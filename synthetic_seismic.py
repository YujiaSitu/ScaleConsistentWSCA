import math
import torch
import torch.nn.functional as F


def soft_resample_logs_gaussian(
    well_data: torch.Tensor, well_mask: torch.Tensor, z_query: torch.Tensor, sigma: float = 0.20, eps: float = 1e-12
):
    """Resample (B, L, C) logs at (B, N) query depths using Gaussian weights."""

    assert well_data.dim() == 3 and z_query.dim() == 2
    B, L, C = well_data.shape
    _, N = z_query.shape

    dtype = well_data.dtype

    z_log = well_data[:, :, 0].to(dtype)

    valid_depth = (well_mask > 0).any(dim=-1).to(dtype)

    diff = z_query.unsqueeze(-1) - z_log.unsqueeze(1)

    w = torch.exp(-0.5 * (diff / float(sigma)) ** 2)

    w = w * valid_depth.unsqueeze(1)

    w_sum = w.sum(dim=-1, keepdim=True).clamp_min(eps)
    w_norm = w / w_sum

    out = torch.bmm(w_norm, well_data)

    return out


def _linear_interp_1d_xfixed(xq: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
    """
    x: (K,)
    y: (B,K) or (K,)
    xq: (M,)
    return: (B,M)
    """
    if y.dim() == 1:
        y = y.unsqueeze(0)
    B, K = y.shape

    xq = xq.clamp(min=x[0], max=x[-1])
    idx = torch.searchsorted(x, xq, right=False).clamp(1, K - 1)
    i0 = idx - 1
    i1 = idx

    x0 = x[i0]
    x1 = x[i1]
    w = (xq - x0) / (x1 - x0 + 1e-12)

    y0 = y[:, i0]
    y1 = y[:, i1]
    return y0 + (y1 - y0) * w.unsqueeze(0)


def Synthetic_seismic_torch(
    well_data: torch.Tensor,
    depth_grid: torch.Tensor,
    pre_time_depth: torch.Tensor,
    dt_seis: float = 0.002,
    nt_seis: int = 501,
    f0: float = 17.0,
    DEN_COL: int = 6,
    DTC_COL: int = 8,
    DEPTH_COL: int = 0,
    eps: float = 1e-12,
):
    """Synthesize (B, nt) traces from density, sonic logs and depth/time pairs."""
    device = well_data.device
    dtype = well_data.dtype

    z_log = well_data[:, DEPTH_COL].to(dtype)
    den = well_data[:, DEN_COL].to(dtype)
    dtc = well_data[:, DTC_COL].to(dtype)

    valid = (den > 0) & (dtc > 0)

    vp = torch.zeros_like(dtc)
    vp[valid] = 1.0 / (dtc[valid] * 1e-6 / 0.3048 + eps)

    rho = torch.zeros_like(den)
    rho[valid] = den[valid] * 1000.0

    Z = vp * rho

    Z1, Z0 = Z[1:], Z[:-1]
    valid_pair = valid[1:] & valid[:-1]
    r = torch.zeros_like(Z1)
    r[valid_pair] = (Z1[valid_pair] - Z0[valid_pair]) / (Z1[valid_pair] + Z0[valid_pair] + eps)
    z_mid = 0.5 * (z_log[1:] + z_log[:-1])

    if pre_time_depth.dim() == 2:
        pre_time_depth = pre_time_depth.unsqueeze(0)
    B = pre_time_depth.shape[0]
    t_grid = pre_time_depth[:, :, 1].to(dtype).clone()
    t_grid[:, 0] = 0.0

    x = depth_grid.to(dtype).to(device)

    t_mid = _linear_interp_1d_xfixed(z_mid.to(dtype), x, t_grid)

    M = r.numel()
    r_expand = r.unsqueeze(0).expand(B, M)

    pos = (t_mid / dt_seis).clamp(0.0, nt_seis - 2.000001)

    i0 = torch.floor(pos)
    w = (pos - i0).clamp(0.0, 1.0)
    i0 = i0.to(torch.long)
    i1 = i0 + 1

    r_t = torch.zeros((B, nt_seis), device=device, dtype=dtype)
    r0 = r_expand * (1.0 - w)
    r1 = r_expand * w
    r_t.scatter_add_(dim=1, index=i0, src=r0)
    r_t.scatter_add_(dim=1, index=i1, src=r1)

    t = torch.arange(nt_seis, device=device, dtype=dtype) * dt_seis
    t0 = (nt_seis // 2) * dt_seis
    xw = math.pi * f0 * (t - t0)
    wavelet = (1.0 - 2.0 * xw**2) * torch.exp(-(xw**2))

    inp = r_t.unsqueeze(1)
    ker = wavelet.flip(0).view(1, 1, -1)
    pad = nt_seis // 2
    out = F.conv1d(inp, ker, padding=pad)
    out = out[:, 0, :nt_seis]

    return out
