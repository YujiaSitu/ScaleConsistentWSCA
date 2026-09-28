import argparse
import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from tqdm import tqdm
from dataset import prepare_well_tensors
import random
from pathlib import Path
import os
import scipy.io as sio


def fold_train_test_masks(mask_y_1d, train_idx, test_idx, L, device):
    """Map fold indices onto valid labeled depth positions."""
    valid_idx = torch.where(mask_y_1d)[0].detach().cpu().numpy()
    train_pos = valid_idx[train_idx]
    test_pos = valid_idx[test_idx]

    mask_train = torch.zeros(L, dtype=torch.bool, device=device)
    mask_test = torch.zeros(L, dtype=torch.bool, device=device)
    mask_train[train_pos] = True
    mask_test[test_pos] = True
    return mask_train.unsqueeze(0), mask_test.unsqueeze(0)


def compute_vmax_99(section):
    vmax = np.percentile(np.abs(section), 99.0)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return float(vmax)


def masked_mse(pred, target, mask):
    while mask.dim() < pred.dim():
        mask = mask.unsqueeze(-1)
    mask_f = mask.to(pred.dtype)

    diff2 = (pred - target) ** 2
    diff2 = diff2 * mask_f

    denom = mask_f.sum().clamp_min(1.0)

    if pred.dim() >= 1 and pred.size(-1) > 1 and mask_f.size(-1) == 1:
        denom = denom * pred.size(-1)

    return diff2.sum() / denom


def regression_metrics_mask3(y_pred: torch.Tensor, y_true: torch.Tensor, mask3: torch.Tensor, eps: float = 1e-12):
    """Compute per-target MAE, RMSE and R2 using channel-specific masks."""
    names = ["POR", "PERM", "SW"]
    out = {}
    maes, rmses, r2s = [], [], []

    for c, name in enumerate(names):
        m = mask3[..., c].reshape(-1)
        yp = y_pred[..., c].reshape(-1)[m].float()
        yt = y_true[..., c].reshape(-1)[m].float()

        if yp.numel() == 0:
            out[name] = {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
            continue

        err = yp - yt
        mae = err.abs().mean()
        rmse = torch.sqrt((err**2).mean() + eps)

        yt_mean = yt.mean()
        sse = (err**2).sum()
        sst = ((yt - yt_mean) ** 2).sum() + eps
        r2 = 1.0 - sse / sst

        out[name] = {"mae": mae.item(), "rmse": rmse.item(), "r2": r2.item()}
        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)

    metrics_mean = {
        "mae": torch.stack(maes).mean().item() if len(maes) else float("nan"),
        "rmse": torch.stack(rmses).mean().item() if len(rmses) else float("nan"),
        "r2": torch.stack(r2s).mean().item() if len(r2s) else float("nan"),
    }
    return out, metrics_mean


def standardize(x, mean, std):
    return (x - mean.view(1, 1, -1)) / std.view(1, 1, -1)


def masked_shift_ncc_loss(pred, target, mask, max_shift=40, tau=0.2, eps=1e-6):
    pred = pred.contiguous()
    target = target.contiguous()
    m = mask.to(pred.dtype)

    B, T = pred.shape
    corrs = []

    for s in range(-max_shift, max_shift + 1):
        if s < 0:
            p = pred[:, : T + s]
            t = target[:, -s:]
            mm = m[:, : T + s] * m[:, -s:]
        elif s > 0:
            p = pred[:, s:]
            t = target[:, : T - s]
            mm = m[:, s:] * m[:, : T - s]
        else:
            p, t, mm = pred, target, m

        wsum = mm.sum(dim=1).clamp_min(1.0)
        pm = (p * mm).sum(dim=1) / wsum
        tm = (t * mm).sum(dim=1) / wsum

        pc = (p - pm[:, None]) * mm
        tc = (t - tm[:, None]) * mm

        num = (pc * tc).sum(dim=1)
        den = torch.sqrt((pc**2).sum(dim=1) + eps) * torch.sqrt((tc**2).sum(dim=1) + eps)
        corr = num / (den + eps)
        corrs.append(corr)

    corrs = torch.stack(corrs, dim=1)
    weights = torch.softmax(corrs / tau, dim=1)
    corr_soft = (weights * corrs).sum(dim=1)
    loss = 1.0 - corr_soft.mean()

    return loss


def masked_smoothl1(pred, target, mask, beta=1.0):

    m = mask.float()
    loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    return (loss * m).sum() / m.sum().clamp_min(1.0)


def regression_mape_mask3(y_pred: torch.Tensor, y_true: torch.Tensor, mask3: torch.Tensor, eps: float = 1e-6):
    """
    MAPE (decimal, NOT percentage)
    MAPE = mean(|yhat - y| / max(|y|, eps))
    """
    names = ["POR", "PERM", "SW"]
    out = {}
    mapes = []

    for c, name in enumerate(names):
        m = mask3[..., c].reshape(-1)
        yp = y_pred[..., c].reshape(-1)[m].float()
        yt = y_true[..., c].reshape(-1)[m].float()

        if yp.numel() == 0:
            out[name] = float("nan")
            continue

        denom = yt.abs().clamp_min(eps)
        mape = ((yp - yt).abs() / denom).mean()
        out[name] = mape.item()
        mapes.append(mape)

    mean_mape = torch.stack(mapes).mean().item() if len(mapes) else float("nan")
    return out, mean_mape


def main(argv=None):
    """Parse arguments and run full-well cross-validation."""
    parser = argparse.ArgumentParser(description="Joint well-log and seismic training")
    parser.add_argument("--n_splits", type=int, default=10, help="Number of KFold splits")
    parser.add_argument(
        "--shuffle", action=argparse.BooleanOptionalAction, default=True, help="Shuffle samples in KFold"
    )
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for KFold")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for repeat")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0092, help="Initial learning rate")
    parser.add_argument("--T0", type=float, default=100.0, help="Initial well time (ms)")
    parser.add_argument("--lambda_recon", type=float, default=0.005, help="rec loss")
    parser.add_argument("--lambda_seis", type=float, default=0.05, help="synthetic seismic loss")
    parser.add_argument("--file_path", type=str, default="well_X.xlsx")
    parser.add_argument("--mask_xlsx_path", type=str, default="well_mask.xlsx")
    parser.add_argument("--segy_path", type=str, default="well.sgy")
    parser.add_argument("--init_time_depth_path", type=str, default="well_time_depth_init.xlsx")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio inside train fold")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/run_1"))
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data" / "synthetic")
    parser.add_argument(
        "--trace-indices",
        type=int,
        nargs=32,
        default=list(range(32)),
        help="Exactly 32 zero-based trace indices, in context order",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    opt = parser.parse_args(argv)
    for field in ("file_path", "mask_xlsx_path", "segy_path", "init_time_depth_path"):
        path = Path(getattr(opt, field))
        if not path.is_absolute():
            path = opt.data_dir / path
        if not path.is_file():
            parser.error(f"Missing input file: {path}")
        setattr(opt, field, path)
    if opt.epochs < 1 or opt.n_splits < 2 or not 0 < opt.val_ratio < 1:
        parser.error("Require epochs >= 1, n_splits >= 2 and 0 < val_ratio < 1.")
    from model import JointModel

    save_dir = opt.output_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    result_txt = save_dir / "cv_results.txt"

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if opt.device == "auto" else opt.device)

    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    trace_indices = opt.trace_indices

    data = prepare_well_tensors(
        device=device,
        trace_indices=trace_indices,
        T0_ms=opt.T0,
        t_max_s=1.0,
        dt_seis=0.002,
        nt_seis=501,
        file_path=opt.file_path,
        mask_xlsx_path=opt.mask_xlsx_path,
        segy_path=opt.segy_path,
        init_time_depth_path=opt.init_time_depth_path,
    )

    well_data_t = data["well_data"]
    y_gt = data["y_gt"]
    mask_y_all = data["mask_y_all"]
    mask_x_all = data["mask_x_all"]
    seis_501_t = data["seis_501"]
    seis_obs = data["seis_obs"]
    mask_time = data["mask_time"]
    depth_grid = data["depth_grid"]
    x_td = data["x_td"]
    y_mask_3 = data["y_mask_3_t"].bool()
    well_mask_t = data["well_mask"]

    B, L, C = well_data_t.shape

    vmax = compute_vmax_99(seis_501_t.detach().cpu().numpy())

    seis_obs = seis_obs.clamp(min=-vmax, max=vmax)
    mask_seis = mask_time.unsqueeze(0).to(device)
    scale_seis = vmax + 1e-12
    seis_obs = seis_obs / scale_seis
    seis_501_t = seis_501_t / scale_seis

    mask_y_1d = mask_y_all.squeeze(0)
    n_valid = int(mask_y_1d.sum().item())
    if n_valid < opt.n_splits:
        raise ValueError(f"Too few valid labeled depths ({n_valid}) for {opt.n_splits} folds.")

    valid_idx_np = torch.where(mask_y_1d)[0].detach().cpu().numpy()
    kf = KFold(n_splits=opt.n_splits, shuffle=opt.shuffle, random_state=opt.random_state if opt.shuffle else None)

    test_mae = np.zeros((opt.n_splits, 3), dtype=np.float64)
    test_rmse = np.zeros((opt.n_splits, 3), dtype=np.float64)
    test_r2 = np.zeros((opt.n_splits, 3), dtype=np.float64)
    test_mape = np.zeros((opt.n_splits, 3), dtype=np.float64)

    splits = []

    for fold, (train_subidx, test_subidx) in enumerate(kf.split(valid_idx_np), start=1):

        _, mask_test = fold_train_test_masks(mask_y_1d, train_subidx, test_subidx, L, device=device)

        train_subidx = np.array(train_subidx)
        rng = np.random.RandomState(opt.seed + fold)
        n_train = len(train_subidx)
        if n_train < 2:
            raise ValueError("Each training fold needs at least two labeled depths.")
        n_val = min(n_train - 1, max(1, int(round(n_train * opt.val_ratio))))
        perm = rng.permutation(n_train)

        val_subidx = train_subidx[perm[:n_val]]
        train_fit_subidx = train_subidx[perm[n_val:]]

        mask_train, mask_val = fold_train_test_masks(mask_y_1d, train_fit_subidx, val_subidx, L, device=device)

        splits.append(
            {
                "fold": fold,
                "train_fit_subidx": train_fit_subidx,
                "val_subidx": val_subidx,
                "test_subidx": test_subidx,
                "mask_train": mask_train,
                "mask_val": mask_val,
                "mask_test": mask_test,
            }
        )

    with open(result_txt, "w", encoding="utf-8") as f:
        f.write(f"{opt.n_splits}-Fold CV Results (Best model selected by VAL mean RMSE)\n")
        f.write("------------------------------------------------------\n")

    for sp in splits:
        fold = sp["fold"]
        mask_train = sp["mask_train"]
        mask_val = sp["mask_val"]
        mask_test = sp["mask_test"]
        print(f"\n========== Fold {fold}/{opt.n_splits} ==========")

        mask_y_train = mask_y_all & mask_train
        mask_y_val = mask_y_all & mask_val
        mask_y_test = mask_y_all & mask_test

        mask_x_train = mask_x_all & mask_train

        m3_train = y_mask_3 & mask_y_train.unsqueeze(-1)
        m3_val = y_mask_3 & mask_y_val.unsqueeze(-1)

        with torch.no_grad():
            xm = well_mask_t.bool() & mask_x_train.unsqueeze(-1)
            xm_f = xm.float()
            cnt = xm_f.sum(dim=(0, 1)).clamp_min(1.0)
            x_mean_in = (well_data_t * xm_f).sum(dim=(0, 1)) / cnt
            x_var = ((well_data_t - x_mean_in) ** 2 * xm_f).sum(dim=(0, 1)) / cnt
            x_std_in = x_var.sqrt().clamp_min(1e-6)

        x_norm = (well_data_t - x_mean_in) / x_std_in
        x_norm = x_norm.clamp(-5.0, 5.0)

        pbar = tqdm(range(opt.epochs), dynamic_ncols=True, smoothing=0.01)
        model = JointModel(depth_grid=depth_grid.detach().cpu(), initial_time_ms=opt.T0).to(device)
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)

        best_val_rmse = float("inf")
        best_path = os.path.join(save_dir, "model{}.pth".format(fold))

        for epoch in pbar:
            model.train()
            optimizer.zero_grad()

            out = model(
                x_td=x_td,
                well_data=x_norm,
                well_raw=well_data_t,
                well_mask=well_mask_t,
                seis_501=seis_501_t,
                depth_grid=depth_grid,
                dt_seis=0.002,
                nt_seis=501,
                f0=17.0,
                sigma=0.20,
                clip_vmax=vmax,
            )
            y_hat = out["y_hat"]
            x_rec = out["x_rec"]
            seis_syn = out["seis_syn"]

            rec_idx = torch.tensor([6, 5, 14, 8], device=well_data_t.device)

            x_gt_rec = well_data_t[..., rec_idx]
            m_rec = well_mask_t[..., rec_idx].bool() & mask_x_train.unsqueeze(-1)

            with torch.no_grad():
                mrf = m_rec.float()
                cnt = mrf.sum(dim=(0, 1)).clamp_min(1.0)
                x_mean_rec = (x_gt_rec * mrf).sum(dim=(0, 1)) / cnt
                x_var_rec = ((x_gt_rec - x_mean_rec) ** 2 * mrf).sum(dim=(0, 1)) / cnt
                x_std_rec = x_var_rec.sqrt().clamp_min(1e-6)

            x_rec_s = standardize(x_rec, x_mean_rec, x_std_rec)
            x_gt_sx = standardize(x_gt_rec, x_mean_rec, x_std_rec)

            loss_recon = masked_mse(x_rec_s, x_gt_sx, m_rec)

            loss_main = 0.0
            for c in range(3):
                if c == 1:
                    lc = masked_smoothl1(y_hat[..., c], y_gt[..., c], m3_train[..., c], beta=1.0)
                else:
                    lc = masked_mse(y_hat[..., c], y_gt[..., c], m3_train[..., c])
                loss_main += lc
            loss_main = loss_main / 3.0

            seis_syn = out["seis_syn"]

            m = mask_seis.bool()
            scale = seis_obs[m].std(unbiased=False).clamp_min(1e-6).detach()

            seis_syn_s = seis_syn / scale
            seis_obs_s = seis_obs / scale

            loss_seis = masked_shift_ncc_loss(seis_syn_s, seis_obs_s, mask_seis)

            total_loss = loss_main + opt.lambda_recon * loss_recon + opt.lambda_seis * loss_seis
            total_loss.backward()
            optimizer.step()

            _, metrics_mean = regression_metrics_mask3(y_hat, y_gt, m3_train)
            _, train_mape_mean = regression_mape_mask3(y_hat, y_gt, m3_train)

            pbar.set_postfix(
                loss_main=f"{loss_main.item():.4f}",
                loss_rec=f"{opt.lambda_recon*loss_recon.item():.4f}",
                loss_seis=f"{opt.lambda_seis *loss_seis.item():.4f}",
                MAE=f"{metrics_mean['mae']:.4f}",
                RMSE=f"{metrics_mean['rmse']:.4f}",
                R2=f"{metrics_mean['r2']:.4f}",
                MAPE=f"{train_mape_mean:.4f}",
            )

            with torch.no_grad():
                model.eval()
                out_v = model(
                    x_td=x_td,
                    well_data=x_norm,
                    well_raw=well_data_t,
                    well_mask=well_mask_t,
                    seis_501=seis_501_t,
                    depth_grid=depth_grid,
                    dt_seis=0.002,
                    nt_seis=501,
                    f0=17.0,
                    sigma=0.20,
                    clip_vmax=vmax,
                )
                y_hat_v = out_v["y_hat"]

                _, val_mean = regression_metrics_mask3(y_hat_v, y_gt, m3_val)
                val_rmse = float(val_mean["rmse"])

                _, val_mape_mean = regression_mape_mask3(y_hat_v, y_gt, m3_val)

                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    torch.save(model.state_dict(), best_path)
                    torch.save(
                        {
                            "x_mean": x_mean_in.detach().cpu(),
                            "x_std": x_std_in.detach().cpu(),
                            "seismic_scale": scale_seis,
                            "initial_time_ms": opt.T0,
                            "trace_indices": trace_indices,
                            "train_mask": mask_train.cpu(),
                            "validation_mask": mask_val.cpu(),
                            "test_mask": mask_test.cpu(),
                        },
                        save_dir / f"preprocessing{fold}.pt",
                    )

                model.train()

        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        model.eval()

        x_norm_test = (well_data_t - x_mean_in) / x_std_in
        x_norm_test = x_norm_test.clamp(-5, 5)

        with torch.no_grad():
            out = model(
                x_td=x_td,
                well_data=x_norm_test,
                well_raw=well_data_t,
                well_mask=well_mask_t,
                seis_501=seis_501_t,
                depth_grid=depth_grid,
                dt_seis=0.002,
                nt_seis=501,
                f0=17.0,
                sigma=0.20,
                clip_vmax=vmax,
            )

            y_hat_test = out["y_hat"]

            m3_test = y_mask_3 & mask_y_test.unsqueeze(-1)

            metrics_var, metrics_mean = regression_metrics_mask3(y_hat_test, y_gt, m3_test)
            mape_var, mape_mean = regression_mape_mask3(y_hat_test, y_gt, m3_test)

            names = ["POR", "PERM", "SW"]
            for i, nm in enumerate(names):
                test_mae[fold - 1, i] = metrics_var[nm]["mae"]
                test_rmse[fold - 1, i] = metrics_var[nm]["rmse"]
                test_r2[fold - 1, i] = metrics_var[nm]["r2"]
                test_mape[fold - 1, i] = mape_var[nm]

            print(
                f"Fold {fold} Test(best) | "
                f"POR(MAE={metrics_var['POR']['mae']:.3f}, RMSE={metrics_var['POR']['rmse']:.3f}, R2={metrics_var['POR']['r2']:.3f}, MAPE={mape_var['POR']:.3f}) | "
                f"PERM(MAE={metrics_var['PERM']['mae']:.3f}, RMSE={metrics_var['PERM']['rmse']:.3f}, R2={metrics_var['PERM']['r2']:.3f}, MAPE={mape_var['PERM']:.3f}) | "
                f"SW(MAE={metrics_var['SW']['mae']:.3f}, RMSE={metrics_var['SW']['rmse']:.3f}, R2={metrics_var['SW']['r2']:.3f}, MAPE={mape_var['SW']:.3f}) | "
                f"MEAN(MAE={metrics_mean['mae']:.3f}, RMSE={metrics_mean['rmse']:.3f}, R2={metrics_mean['r2']:.3f}, MAPE={mape_mean:.3f})"
            )
            sio.savemat(
                str(save_dir / f"y_hat_fold{fold:02d}.mat"), {"y_hat": y_hat_test.squeeze(0).detach().cpu().numpy()}
            )

            with open(result_txt, "a", encoding="utf-8") as f:
                f.write(
                    f"Fold {fold:02d} | "
                    f"POR: MAE={metrics_var['POR']['mae']:.4f}, RMSE={metrics_var['POR']['rmse']:.4f}, R2={metrics_var['POR']['r2']:.4f}, MAPE={mape_var['POR']:.4f} | "
                    f"PERM: MAE={metrics_var['PERM']['mae']:.4f}, RMSE={metrics_var['PERM']['rmse']:.4f}, R2={metrics_var['PERM']['r2']:.4f}, MAPE={mape_var['PERM']:.4f} | "
                    f"SW: MAE={metrics_var['SW']['mae']:.4f}, RMSE={metrics_var['SW']['rmse']:.4f}, R2={metrics_var['SW']['r2']:.4f}, MAPE={mape_var['SW']:.4f} | "
                    f"MEAN: MAE={metrics_mean['mae']:.4f}, RMSE={metrics_mean['rmse']:.4f}, R2={metrics_mean['r2']:.4f}, MAPE={mape_mean:.4f}\n"
                )

    print(f"\n========== {opt.n_splits}-Fold Cross-Validation Performance (BEST by VAL) ==========")

    vars_ = ["POR", "PERM", "SW"]
    for i, v in enumerate(vars_):
        print(
            f"{v:<5} | "
            f"MAE: {test_mae[:, i].mean():.4f} ± {test_mae[:, i].std():.4f} | "
            f"RMSE: {test_rmse[:, i].mean():.4f} ± {test_rmse[:, i].std():.4f} | "
            f"R2: {test_r2[:, i].mean():.4f} ± {test_r2[:, i].std():.4f} | "
            f"MAPE: {test_mape[:, i].mean():.4f} ± {test_mape[:, i].std():.4f}"
        )

    print("-" * 70)

    mean_mae = test_mae.mean(axis=1)
    mean_rmse = test_rmse.mean(axis=1)
    mean_r2 = test_r2.mean(axis=1)
    mean_mape = test_mape.mean(axis=1)

    print(
        f"MEAN  | "
        f"MAE: {mean_mae.mean():.4f} ± {mean_mae.std():.4f} | "
        f"RMSE: {mean_rmse.mean():.4f} ± {mean_rmse.std():.4f} | "
        f"R2: {mean_r2.mean():.4f} ± {mean_r2.std():.4f} | "
        f"MAPE: {mean_mape.mean():.4f} ± {mean_mape.std():.4f}"
    )

    with open(result_txt, "a", encoding="utf-8") as f:
        f.write("\n------------------------------------------------------\n")
        f.write("Summary (mean ± std over folds)\n")
        for i, v in enumerate(vars_):
            f.write(
                f"{v:<5} | "
                f"MAE: {test_mae[:, i].mean():.4f} ± {test_mae[:, i].std():.4f} | "
                f"RMSE: {test_rmse[:, i].mean():.4f} ± {test_rmse[:, i].std():.4f} | "
                f"R2: {test_r2[:, i].mean():.4f} ± {test_r2[:, i].std():.4f} | "
                f"MAPE: {test_mape[:, i].mean():.4f} ± {test_mape[:, i].std():.4f}\n"
            )
        f.write(
            f"MEAN  | "
            f"MAE: {mean_mae.mean():.4f} ± {mean_mae.std():.4f} | "
            f"RMSE: {mean_rmse.mean():.4f} ± {mean_rmse.std():.4f} | "
            f"R2: {mean_r2.mean():.4f} ± {mean_r2.std():.4f} | "
            f"MAPE: {mean_mape.mean():.4f} ± {mean_mape.std():.4f}\n"
        )

    print(f"\n[Saved] best models: {save_dir}/model1.pth ... model{opt.n_splits}.pth")
    print(f"[Saved] results log: {result_txt}")


if __name__ == "__main__":
    main()
