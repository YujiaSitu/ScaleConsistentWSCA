import numpy as np
import pandas as pd
import torch
import segyio


def _to_float32_np(x):
    x = np.asarray(x)
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    return x


def load_well_raw(
    file_path="well_X.xlsx",
    mask_xlsx_path="well_mask.xlsx",
    segy_path="well.sgy",
    init_time_depth_path="well_time_depth_init.xlsx",
    trace_indices=None,
):
    """Load well features, validity masks, seismic traces and initial depth/time pairs."""

    feature_cols = [
        "DEPTH",
        "AC",
        "BRIT",
        "BRITV",
        "CAL",
        "CNL",
        "DEN",
        "DEVI",
        "DTC",
        "DTS",
        "DTST",
        "GAS_FREE",
        "GAS_SORB",
        "GAS_T",
        "GR",
        "HAZI",
        "K",
        "KTH",
        "RT",
        "RXO",
        "TH",
        "TOC",
        "URAN",
        "VOL_CALCITE",
        "VOL_ILLITE",
        "VOL_KEROGEN",
        "VOL_PYRITE",
        "VOL_QUARTZ",
        "VOL_UBNDWAT",
    ]
    target_cols = ["POR", "PERM", "SW"]

    well_df = pd.read_excel(file_path).reset_index(drop=True)
    mask_df = pd.read_excel(mask_xlsx_path).reset_index(drop=True)

    required = feature_cols + target_cols
    for label, frame in (("well data", well_df), ("validity mask", mask_df)):
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns in {label}: {sorted(missing)}")
    if len(well_df) != len(mask_df) or len(well_df) < 2:
        raise ValueError("Well data and masks must have matching rows (at least two).")

    X = well_df[feature_cols].to_numpy(dtype=np.float32)
    Y = well_df[target_cols].to_numpy(dtype=np.float32)

    X_mask = mask_df[feature_cols].to_numpy()
    Y_mask = mask_df[target_cols].to_numpy()

    X_mask = _to_float32_np(X_mask)
    Y_mask = _to_float32_np(Y_mask)

    if not np.isin(X_mask, [0, 1]).all() or not np.isin(Y_mask, [0, 1]).all():
        raise ValueError("Validity masks must contain only 0 and 1.")
    if not np.isfinite(X[X_mask > 0]).all() or not np.isfinite(Y[Y_mask > 0]).all():
        raise ValueError("Valid data entries must be finite.")
    if not np.isfinite(X[:, 0]).all() or not (np.diff(X[:, 0]) > 0).all():
        raise ValueError("DEPTH must be finite and strictly increasing.")
    if np.any(Y[Y_mask[:, 1] > 0, 1] < 0):
        raise ValueError("Valid PERM values must be nonnegative.")

    X = np.nan_to_num(X, nan=0.0).astype(np.float32)
    Y = np.nan_to_num(Y, nan=0.0).astype(np.float32)

    perm_idx = target_cols.index("PERM")

    perm = Y[:, perm_idx]
    perm_mask = Y_mask[:, perm_idx] > 0

    perm_log = np.zeros_like(perm, dtype=np.float32)
    perm_log[perm_mask] = np.log(perm[perm_mask] + 1e-6)

    perm_log = np.clip(perm_log, -10.0, 10.0)

    Y[:, perm_idx] = perm_log

    with segyio.open(segy_path, "r", ignore_geometry=True) as f:
        f.mmap()
        if len(f.samples) != 501 or not np.isclose(segyio.tools.dt(f), 2000):
            raise ValueError("Expected 501 SEG-Y samples and a 2000-microsecond interval.")
        if trace_indices is None:
            seismic_np = segyio.tools.collect(f.trace[:])
        else:
            if len(trace_indices) != 32 or min(trace_indices) < 0 or max(trace_indices) >= f.tracecount:
                raise ValueError("Provide 32 trace indices within the SEG-Y trace range.")
            seismic_np = np.stack([f.trace[index].copy() for index in trace_indices])
    seismic_np = _to_float32_np(seismic_np)
    if not np.isfinite(seismic_np).all():
        raise ValueError("Selected seismic traces must be finite.")

    td_df = pd.read_excel(init_time_depth_path)
    if len(td_df) != 16 or not {"DEPTH", "TWT_ms_init"}.issubset(td_df.columns):
        raise ValueError("Initial time-depth data requires 16 rows and DEPTH/TWT_ms_init columns.")
    depth = td_df["DEPTH"].to_numpy(dtype=np.float32)
    time_ms = td_df["TWT_ms_init"].to_numpy(dtype=np.float32)
    if not np.isfinite(depth).all() or not np.isfinite(time_ms).all():
        raise ValueError("Initial time-depth entries must be finite.")

    time_s = time_ms * 1e-3
    time_depth_mat = np.stack([depth, time_s], axis=1).astype(np.float32)
    time_mask_td = (time_s > 0).astype(np.float32)

    return (X, X_mask, Y, Y_mask, seismic_np, time_depth_mat, time_mask_td, feature_cols, target_cols)


def prepare_well_tensors(
    device,
    trace_indices,
    T0_ms,
    t_max_s=1.0,
    dt_seis=0.002,
    nt_seis=501,
    file_path="well_X.xlsx",
    mask_xlsx_path="well_mask.xlsx",
    segy_path="well.sgy",
    init_time_depth_path="well_time_depth_init.xlsx",
):
    """Prepare full-well tensors and masks on the requested device."""

    (X, X_mask, Y, Y_mask, seismic_np, time_depth_mat, time_mask_td, feature_cols, target_cols) = load_well_raw(
        file_path=file_path,
        mask_xlsx_path=mask_xlsx_path,
        segy_path=segy_path,
        init_time_depth_path=init_time_depth_path,
        trace_indices=trace_indices,
    )

    depth_grid_np = np.linspace(X[0, 0], X[-1, 0], nt_seis).astype(np.float32)

    seis_501_np = seismic_np.T
    seis_501_t = torch.tensor(seis_501_np, dtype=torch.float32, device=device).unsqueeze(0)

    mid_k = len(trace_indices) // 2
    seis_obs_t = seis_501_t[:, :, mid_k]

    T0_s = float(T0_ms) * 1e-3
    t = torch.arange(nt_seis, device=device, dtype=torch.float32) * float(dt_seis)
    mask_time = (t >= T0_s) & (t <= float(t_max_s))
    if not mask_time.any():
        raise ValueError("The selected seismic time window contains no samples.")

    well_data_t = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(0)
    well_mask_t = torch.tensor(X_mask, dtype=torch.float32, device=device).unsqueeze(0)

    y_gt_t = torch.tensor(Y, dtype=torch.float32, device=device).unsqueeze(0)
    y_mask_t = torch.tensor(Y_mask, dtype=torch.float32, device=device).unsqueeze(0)

    mask_y_all = (y_mask_t > 0).any(dim=-1)
    y_mask_3_t = torch.tensor(Y_mask, dtype=torch.float32, device=device).unsqueeze(0)

    idx_den = feature_cols.index("DEN")
    idx_cnl = feature_cols.index("CNL")
    idx_gr = feature_cols.index("GR")
    idx_dtc = feature_cols.index("DTC")
    x_idx = [idx_den, idx_cnl, idx_gr, idx_dtc]

    x_gt_t = well_data_t[:, :, x_idx]
    x_mask_t = (well_mask_t[:, :, x_idx] > 0).all(dim=-1)

    x_td = torch.tensor(time_depth_mat, dtype=torch.float32, device=device).unsqueeze(0)

    depth_grid = torch.tensor(depth_grid_np, dtype=torch.float32, device=device)

    return {
        "well_data": well_data_t,
        "well_mask": well_mask_t,
        "y_gt": y_gt_t,
        "y_mask_3_t": y_mask_3_t,
        "mask_y_all": mask_y_all,
        "x_gt": x_gt_t,
        "mask_x_all": x_mask_t,
        "seis_501": seis_501_t,
        "seis_obs": seis_obs_t,
        "mask_time": mask_time,
        "depth_grid": depth_grid,
        "x_td": x_td,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "time_mask_td": torch.tensor(time_mask_td, dtype=torch.float32, device=device),
    }
