import torch
import torch.nn as nn
import torch.nn.functional as F
from synthetic_seismic import Synthetic_seismic_torch, soft_resample_logs_gaussian


class TDNet(nn.Module):
    def __init__(self, depth_grid, num_heads=4):
        super().__init__()
        self.hid1 = 64
        self.hid2 = 512
        self.out_len = 501
        self.register_buffer("depth_grid", depth_grid.view(self.out_len, 1).float(), persistent=True)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=self.hid1, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=self.hid1, out_channels=self.hid2, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.attn = nn.MultiheadAttention(embed_dim=self.hid2, num_heads=num_heads, batch_first=True)

        self.delta_fc = nn.Linear(self.hid2, self.out_len - 1)

    def forward(self, x):
        B = x.shape[0]
        feat = self.conv(x)
        feat = feat.transpose(1, 2)
        feat, _ = self.attn(feat, feat, feat)
        g = feat.mean(dim=1)
        Tmax = 0.8

        delta = self.delta_fc(g)
        delta = F.softplus(delta)

        delta_sum = delta.sum(dim=1, keepdim=True) + 1e-8
        delta = delta / delta_sum * Tmax

        t0 = torch.zeros(B, 1, device=delta.device, dtype=delta.dtype)
        t_rest = torch.cumsum(delta, dim=1)
        t = torch.cat([t0, t_rest], dim=1)

        depth = self.depth_grid.to(t.device, t.dtype).unsqueeze(0).expand(B, -1, -1)
        y = torch.cat([depth, t.unsqueeze(-1)], dim=-1)
        return y


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        """
        query: (B, Lq, D)
        key/value: (B, Lk, D)
        """
        attn_out, _ = self.attn(query, key, value, key_padding_mask=key_padding_mask)
        out = self.norm(query + self.dropout(attn_out))
        return out


class CALSTM(nn.Module):
    def __init__(self, raw_log_dim, ctx_log_dim, ctx_seis_dim, ctx_td_dim, hidden_dim, lstm_layers, num_heads):
        super().__init__()

        self.raw_embed = nn.Linear(raw_log_dim, hidden_dim)

        self.ctx_k = nn.Linear(ctx_log_dim + ctx_seis_dim + ctx_td_dim, hidden_dim)
        self.ctx_v = nn.Linear(ctx_log_dim + ctx_seis_dim + ctx_td_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True, bidirectional=True
        )
        self.proj = nn.Linear(2 * hidden_dim, hidden_dim)

        self.cross_ctx_1 = CrossAttention(hidden_dim, num_heads)
        self.cross_ctx_2 = CrossAttention(hidden_dim, num_heads)

        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2), nn.Linear(hidden_dim, 3))
        self.mtrd = MTRD(y_dim=3, hidden_dim=32, shared_layers=2, dropout=0.0)

    def forward(self, logs_raw, logs_501, seis_501, td_501):
        """
        logs_raw: (B, Lraw=10035, 29)
        logs_501: (B, Lctx=501, 29)
        seis_501: (B, Lctx=501, 11)
        td_501:   (B, Lctx=501, 2)
        """

        q0 = self.raw_embed(logs_raw)
        q, _ = self.lstm(q0)
        q = self.proj(q)
        q = q + q0

        ctx = torch.cat([logs_501, seis_501, td_501], dim=-1)
        K = self.ctx_k(ctx)
        V = self.ctx_v(ctx)

        h = q + self.cross_ctx_1(q, K, V)
        h = h + self.cross_ctx_2(h, K, V)

        out = self.head(h)
        rec = self.mtrd(out)
        return out, rec


class MTRD(nn.Module):
    """Reconstruct DEN, CNL, GR and DTC from predicted POR, log(PERM) and SW."""

    def __init__(self, y_dim=3, hidden_dim=32, shared_layers=2, dropout=0.0):
        super().__init__()

        layers = []
        in_dim = y_dim
        for _ in range(shared_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            in_dim = hidden_dim
        self.shared = nn.Sequential(*layers)

        self.head_den = nn.Linear(hidden_dim, 1)
        self.head_cnl = nn.Linear(hidden_dim, 1)
        self.head_gr = nn.Linear(hidden_dim, 1)
        self.head_dtc = nn.Linear(hidden_dim, 1)

    def forward(self, y_hat):
        h = self.shared(y_hat)

        den = self.head_den(h)
        cnl = self.head_cnl(h)
        gr = self.head_gr(h)
        dtc = self.head_dtc(h)

        x_hat = torch.cat([den, cnl, gr, dtc], dim=-1)
        return x_hat


class JointModel(nn.Module):
    """Combine time-depth estimation, log resampling and seismic/log prediction."""

    def __init__(
        self,
        depth_grid: torch.Tensor,
        td_num_heads=2,
        raw_log_dim=29,
        ctx_log_dim=29,
        ctx_seis_dim=32,
        ctx_td_dim=2,
        hidden_dim=128,
        lstm_layers=1,
        num_heads=4,
        default_sigma=0.20,
        initial_time_ms=100.0,
    ):
        super().__init__()
        self.tdnet = TDNet(depth_grid=depth_grid, num_heads=td_num_heads)
        self.main = CALSTM(
            raw_log_dim=raw_log_dim,
            ctx_log_dim=ctx_log_dim,
            ctx_seis_dim=ctx_seis_dim,
            ctx_td_dim=ctx_td_dim,
            hidden_dim=hidden_dim,
            lstm_layers=lstm_layers,
            num_heads=num_heads,
        )
        self.default_sigma = float(default_sigma)
        self.initial_time_s = float(initial_time_ms) * 1e-3
        self.seis_gain = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        x_td: torch.Tensor,
        well_data: torch.Tensor,
        well_raw: torch.Tensor,
        well_mask: torch.Tensor,
        seis_501: torch.Tensor,
        depth_grid: torch.Tensor,
        dt_seis=0.002,
        nt_seis=501,
        f0=17.0,
        sigma=None,
        clip_vmax=None,
    ):
        """Run the forward pass; tensor dimensions follow the enclosing model."""

        td_raw = self.tdnet(x_td)
        B = td_raw.shape[0]

        z = depth_grid.view(1, -1).expand(B, -1)

        raw = td_raw[:, :, 1]
        raw = raw + self.initial_time_s

        td_501 = torch.stack([z, raw], dim=-1)

        z_query = td_501[:, :, 0]
        sig = self.default_sigma if sigma is None else float(sigma)
        logs_501 = soft_resample_logs_gaussian(well_data=well_raw, well_mask=well_mask, z_query=z_query, sigma=sig)

        B = well_raw.shape[0]
        seis_list = []
        for b in range(B):
            s = Synthetic_seismic_torch(
                well_data=well_raw[b],
                depth_grid=depth_grid,
                pre_time_depth=td_501[b : b + 1],
                dt_seis=dt_seis,
                nt_seis=nt_seis,
                f0=f0,
            )
            seis_list.append(s)
        seis_syn = torch.cat(seis_list, dim=0)

        if clip_vmax is not None:
            vmax = float(clip_vmax)
            seis_syn = vmax * torch.tanh(seis_syn / (vmax + 1e-12))
        seis_syn = self.seis_gain * seis_syn

        logs_raw = well_data * well_mask.float()

        y_hat, x_rec = self.main(logs_raw=logs_raw, logs_501=logs_501, seis_501=seis_501, td_501=td_501)

        return {
            "y_hat": y_hat,
            "x_rec": x_rec,
            "td_501": td_501,
            "logs_501": logs_501,
            "seis_syn": seis_syn,
            "td_raw": td_raw,
        }
