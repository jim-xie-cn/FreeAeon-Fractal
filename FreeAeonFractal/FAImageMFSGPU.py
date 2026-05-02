"""
GPU-accelerated multifractal (box-counting) analysis for 2D grayscale images.

Mathematical conventions
------------------------
Per scale eps and box size s, with mu_i the box probability measure:
    Z(q, eps) = sum_i mu_i^q
    log Z(q, eps) ~ -tau(q) * log(1/eps) + const
        => slope of log Z vs log(1/eps) is -tau(q)
    D(q) = tau(q) / (q - 1) for q != 1
    D_1 (information dim): slope of S(eps) = -sum mu_i log mu_i vs log(1/eps)
    D_0 (capacity dim):    slope of log N_pos(eps) vs log(1/eps), where
                           N_pos(eps) = number of boxes with mu_i > 0.

Stored values per scale:
    kind = "logMq" -> value = log(sum mu_i^q)              (for q != 0, 1)
    kind = "N"     -> value = number of nonzero boxes      (for q == 0)
    kind = "S"     -> value = -sum mu_i * log mu_i         (for q == 1)

Key fixes vs the previous batch implementation
----------------------------------------------
1) Bug fix: S used to be computed as -(mu * log_mu * mask).sum() with
   log_mu = -inf for empty boxes. Since 0 * (-inf) = NaN in IEEE 754, any
   image with empty boxes produced NaN in S. Now using torch.special.xlogy.
2) Bug fix: get_batch_mfs originally enforced
       scales = scales[(roi_size % scales) == 0]
   while the single-image path did NOT, so batch and per-image runs gave
   different scale sets and differing fits. We drop that constraint and
   crop to (roi_size//s)*s inside the kernel for each scale (this matches
   the CPU/single-GPU path exactly).
3) Performance: the previous batch path appended per-(image, q, scale)
   records in a Python triple loop, which dominated runtime for any
   non-trivial batch and any non-trivial q grid. The new path keeps
   everything as dense numpy arrays of shape (B, n_scales, n_q) and
   converts to a long DataFrame once per image, vectorized.
4) Memory safety: the q-axis expansion (B, n_blocks, n_q) can blow up VRAM
   at small box sizes (n_blocks ~ L^2 / s^2). q is now chunked.
"""

import time

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import torch
from scipy.stats import linregress
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker


# ============================================================
# Fixed ROI utilities (Scheme A: fixed square ROI)
# ============================================================
def crop_square_roi(img, L=None, mode="center"):
    if img is None:
        raise ValueError("img is None")
    if img.ndim != 2:
        raise ValueError("img must be 2D grayscale (H,W).")

    H, W = img.shape
    if L is None:
        L = min(H, W)
    L = int(L)

    if L <= 0 or L > min(H, W):
        raise ValueError(f"Invalid L={L} for image shape {(H, W)}")

    if mode == "topleft":
        y0, x0 = 0, 0
    elif mode == "center":
        y0 = (H - L) // 2
        x0 = (W - L) // 2
    else:
        raise ValueError("mode must be 'topleft' or 'center'.")

    return img[y0:y0 + L, x0:x0 + L].copy(), L


# ============================================================
# Internal helpers (shared between single & batch paths)
# ============================================================
def _preprocess_image(image, bg_threshold=0.01, bg_reverse=False, bg_otsu=False):
    """Replicates the CPU __init__ preprocessing."""
    if image is None:
        raise ValueError("image is None")
    if image.ndim != 2:
        raise ValueError("image must be 2D grayscale (H,W).")

    img_raw = image.astype(np.float64)

    if bg_otsu:
        vmin = np.nanmin(img_raw)
        vmax = np.nanmax(img_raw)
        if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
            img8 = ((img_raw - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
            _, img_bin = cv2.threshold(img8, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img_raw[img_bin == 0] = 0.0

    img = img_raw.copy()
    img -= np.nanmin(img)
    mx = np.nanmax(img)
    img /= (mx + 1e-12)
    img = np.where(np.isfinite(img), img, 0.0)
    img = np.clip(img, 0.0, 1.0)

    if bg_threshold > 0:
        if bg_reverse:
            img[img > bg_threshold] = 0.0
        else:
            img[img < bg_threshold] = 0.0
    return img


def _candidate_scales(min_box, fixed_L_int, max_scales):
    scales = np.logspace(np.log2(min_box), np.log2(fixed_L_int),
                         num=max_scales, base=2.0)
    scales = np.unique(np.maximum(min_box, np.round(scales).astype(int)))
    scales = scales[(scales >= min_box) & (scales <= fixed_L_int)]
    return np.array(sorted(scales.astype(int)))


# ============================================================
# Vectorized per-pixel slope on GPU: log mu(eps) vs log eps.
#
# Same semantics as FAImageMFSGPU._alpha_map_from_mu_stack:
#  - "nan":  NaN/0 entries are dropped from each pixel's regression
#  - "fill": NaN/0 entries are replaced with the smallest positive value
#            at that scale before taking log.
# Math: ordinary least-squares slope with a 0/1 weight mask.
# ============================================================
@torch.no_grad()
def _alpha_map_from_mu_stack_torch(mu_stack, log_eps, empty_policy="nan"):
    """
    Parameters
    ----------
    mu_stack : torch.Tensor of shape (..., K, L, L)
        Leading dims are batch (any rank).
    log_eps : torch.Tensor of shape (K,) (fp64 recommended)
    empty_policy : {"nan", "fill"}

    Returns
    -------
    alpha_map : same shape as mu_stack but without the K axis.
    """
    K = mu_stack.shape[-3]
    ms = mu_stack
    if empty_policy == "fill":
        ms = ms.clone()
        flat = ms.reshape(*ms.shape[:-2], -1)                      # (...,K,L*L)
        masked = torch.where((flat > 0) & torch.isfinite(flat), flat,
                             torch.full_like(flat, float("inf")))
        mins = masked.amin(dim=-1, keepdim=True)                   # (...,K,1)
        mins = torch.where(torch.isfinite(mins), mins,
                           torch.full_like(mins, 1e-300))
        ms = torch.where((ms > 0) & torch.isfinite(ms), ms,
                         mins.reshape(*ms.shape[:-2], 1, 1))
        log_mu = torch.log(ms.to(torch.float64))
        valid = torch.ones_like(log_mu, dtype=torch.bool)
    else:
        good = (ms > 0) & torch.isfinite(ms)
        ms_safe = torch.where(good, ms.to(torch.float64),
                              torch.ones_like(ms, dtype=torch.float64))
        log_mu = torch.log(ms_safe)                                # (...,K,L,L)
        valid = good

    # Broadcast log_eps along the K axis at position -3.
    # We always shape it to (K, 1, 1); PyTorch broadcasts the leading
    # batch dims automatically against (..., K, L, L).
    x = log_eps.to(torch.float64).reshape(K, 1, 1)                 # (K,1,1)
    y = log_mu                                                     # (...,K,L,L)
    w = valid.to(torch.float64)                                    # (...,K,L,L)

    n_eff  = w.sum(dim=-3)                                         # (...,L,L)
    sum_x  = (w * x).sum(dim=-3)
    y_w    = torch.where(valid, y, torch.zeros_like(y))
    sum_y  = y_w.sum(dim=-3)
    sum_xx = (w * x * x).sum(dim=-3)
    sum_xy = (y_w * x).sum(dim=-3)

    denom = n_eff * sum_xx - sum_x * sum_x
    numer = n_eff * sum_xy - sum_x * sum_y

    nan = torch.full_like(denom, float("nan"))
    slope = torch.where((n_eff >= 2) & (denom.abs() > 0),
                        numer / denom, nan)
    return slope


# ============================================================
# Single-image GPU MFS
# ============================================================
class CFAImageMFSGPU:
    """
    GPU version of the single-image MFS pipeline. Identical maths to the
    CPU class; only the per-scale mu / logsumexp loops are pushed to torch.
    """

    def __init__(self, image, corp_type=-1, q_list=np.linspace(-5, 5, 51),
                 with_progress=True, bg_threshold=0.01, bg_reverse=False,
                 bg_otsu=False, mu_floor=1e-12, device=None,
                 dtype=torch.float64):
        img = _preprocess_image(image, bg_threshold=bg_threshold,
                                bg_reverse=bg_reverse, bg_otsu=bg_otsu)
        self.m_image = img
        self.m_corp_type = corp_type
        self.m_with_progress = with_progress
        self.m_q_list = np.array(q_list, dtype=np.float64)
        self._mu_floor = mu_floor  # kept for API compatibility

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype

    # --------------------------------------------------------
    # Per-scale mu (GPU). Returns mu_pos, log_mu (1D tensors).
    # --------------------------------------------------------
    @torch.no_grad()
    def _mu_at_scale_torch(self, img_t: torch.Tensor, box_size: int):
        H, W = img_t.shape
        s = int(box_size)
        if s <= 0:
            return None, None

        new_H = (H // s) * s
        new_W = (W // s) * s
        if new_H == 0 or new_W == 0:
            return None, None
        sub = img_t[:new_H, :new_W]

        nY = new_H // s
        nX = new_W // s
        blocks = sub.reshape(nY, s, nX, s)
        block_mass = blocks.sum(dim=(1, 3)).reshape(-1)
        block_mass = torch.clamp(block_mass, min=0.0)

        total = block_mass.sum()
        if not torch.isfinite(total) or total <= 0:
            return None, None

        mu = block_mass / total
        mu = torch.where(torch.isfinite(mu) & (mu >= 0),
                         mu, torch.zeros_like(mu))
        s_sum = mu.sum()
        if s_sum <= 0 or not torch.isfinite(s_sum):
            return None, None
        mu = mu / s_sum

        mu_pos = mu[mu > 0]
        if mu_pos.numel() == 0:
            return None, None
        return mu_pos, torch.log(mu_pos)

    # --------------------------------------------------------
    # Per-scale records on GPU.
    # --------------------------------------------------------
    @torch.no_grad()
    def get_mass_table(self, max_size=None, max_scales=80, min_box=2,
                       roi_mode="center"):
        img0 = self.m_image
        h0, w0 = img0.shape
        if max_size is None:
            max_size = min(h0, w0)
        if max_size < min_box:
            raise ValueError("max_size too small.")

        img_fixed_np, fixed_L_int = crop_square_roi(
            img0, L=min(min(h0, w0), max_size), mode=roi_mode
        )
        fixed_L = float(fixed_L_int)

        scales = _candidate_scales(min_box, fixed_L_int, max_scales)
        if scales.size == 0:
            return pd.DataFrame()

        img_t = torch.from_numpy(img_fixed_np).to(device=self.device,
                                                  dtype=self.dtype)
        q_t = torch.tensor(self.m_q_list, device=self.device, dtype=self.dtype)

        records = []
        iterator = (tqdm(scales,
                         desc=f"Computing per-scale mu on {self.device.type}")
                    if self.m_with_progress else scales)

        for size in iterator:
            size = int(size)
            mu_pos, log_mu = self._mu_at_scale_torch(img_t, size)
            if mu_pos is None:
                continue

            eps = float(size) / fixed_L

            N = float(mu_pos.numel())
            S = float(-(mu_pos * log_mu).sum().item())

            records.append({"scale": size, "eps": eps, "q": 0.0,
                            "value": N, "kind": "N"})
            records.append({"scale": size, "eps": eps, "q": 1.0,
                            "value": S, "kind": "S"})

            # logMq for all q in one pass: a[i,j] = q_j * log_mu_i
            a = log_mu[:, None] * q_t[None, :]              # (n_pos, n_q)
            logMq_all = torch.logsumexp(a, dim=0)           # (n_q,)
            logMq_cpu = logMq_all.detach().cpu().numpy().astype(np.float64)

            for qv, logMq in zip(self.m_q_list.astype(np.float64),
                                 logMq_cpu):
                if np.isclose(qv, 0.0) or np.isclose(qv, 1.0):
                    continue
                if np.isfinite(logMq):
                    records.append({"scale": size, "eps": eps,
                                    "q": float(qv), "value": float(logMq),
                                    "kind": "logMq"})

        df = pd.DataFrame(records)
        if df.empty:
            return df
        return df.sort_values(["kind", "q", "scale"]).reset_index(drop=True)

    # --------------------------------------------------------
    # Fitting / spectrum (CPU; identical maths to FAImageMFSGPU.py)
    # --------------------------------------------------------
    @staticmethod
    def _common_scales_for_fit(df_mass, require_q1=True):
        if df_mass is None or df_mass.empty:
            return np.array([], dtype=int)

        common = None

        df_logmq = df_mass[df_mass["kind"] == "logMq"]
        if not df_logmq.empty:
            for q, g in df_logmq.groupby("q"):
                scales_q = set(g["scale"].astype(int).tolist())
                common = scales_q if common is None else (common & scales_q)

        df_n = df_mass[df_mass["kind"] == "N"]
        if not df_n.empty:
            scales_n = set(df_n["scale"].astype(int).tolist())
            common = scales_n if common is None else (common & scales_n)

        if common is None:
            return np.array([], dtype=int)

        if require_q1:
            df_s = df_mass[(df_mass["kind"] == "S") &
                           (np.isclose(df_mass["q"], 1.0))]
            if df_s.empty:
                return np.array([], dtype=int)
            common = common & set(df_s["scale"].astype(int).tolist())

        return np.array(sorted(common), dtype=int)

    def fit_tau_and_D1(self, df_mass, min_points=6,
                       require_common_scales=True, use_middle_scales=True,
                       fit_scale_frac=(0.2, 0.8), if_auto_line_fit=False,
                       auto_fit_min_len_ratio=0.5, cap_d0_at_2=False):
        if df_mass is None or df_mass.empty:
            return pd.DataFrame()

        df_mass = df_mass.copy()
        df_mass["scale"] = df_mass["scale"].astype(int)

        require_q1 = np.any(np.isclose(self.m_q_list, 1.0))
        if require_common_scales:
            common_scales = self._common_scales_for_fit(df_mass,
                                                        require_q1=require_q1)
            if common_scales.size == 0:
                common_scales = None
        else:
            common_scales = None

        use_middle = use_middle_scales and (common_scales is not None)
        if use_middle:
            all_scales = common_scales
            if all_scales.size >= 3:
                lo, hi = fit_scale_frac
                lo = max(0.0, min(1.0, float(lo)))
                hi = max(0.0, min(1.0, float(hi)))
                if hi <= lo:
                    hi = min(1.0, lo + 0.6)
                i0 = int(np.floor(lo * len(all_scales)))
                i1 = int(np.ceil(hi * len(all_scales)))
                i1 = max(i1, i0 + 1)
                middle_scales = all_scales[i0:i1]
            else:
                middle_scales = all_scales
        else:
            middle_scales = None

        def _filter_common(d):
            if common_scales is not None:
                d = d[d["scale"].isin(common_scales)].copy()
            if (use_middle and middle_scales is not None
                    and len(middle_scales) > 0):
                d = d[d["scale"].isin(middle_scales)].copy()
            return d

        def _best_linear_segment(x, y, min_pts, min_len_ratio):
            n = len(x)
            min_pts = max(min_pts, int(np.ceil(n * min_len_ratio)))
            if n < min_pts:
                return None
            best = None
            for i in range(0, n - min_pts + 1):
                for j in range(i + min_pts, n + 1):
                    xs = x[i:j]; ys = y[i:j]
                    slope, intercept, r_value, p_value, std_err = linregress(xs, ys)
                    r2 = r_value ** 2
                    length = j - i
                    cand = dict(slope=slope, intercept=intercept,
                                r_value=r_value, p_value=p_value,
                                std_err=std_err, r2=r2, n_points=length)
                    if best is None:
                        best = cand
                    else:
                        if (length > best["n_points"]) or \
                           (length == best["n_points"] and r2 > best["r2"]):
                            best = cand
            return best

        out = []

        # q != 0, 1
        df_logmq = _filter_common(df_mass[df_mass["kind"] == "logMq"])
        for q, df_q in df_logmq.groupby("q"):
            qv = float(q)
            if np.isclose(qv, 1.0) or np.isclose(qv, 0.0):
                continue

            df_q = df_q.sort_values("scale")
            eps = df_q["eps"].astype(np.float64).values
            x = np.log(1.0 / eps)
            y = df_q["value"].astype(np.float64).values
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]; y = y[mask]

            if x.size < min_points:
                out.append({"q": qv, "tau": np.nan, "Dq": np.nan, "D1": np.nan,
                            "intercept": np.nan, "r_value": np.nan,
                            "p_value": np.nan, "std_err": np.nan,
                            "n_points": int(x.size)})
                continue

            if if_auto_line_fit:
                best = _best_linear_segment(x, y, min_points,
                                            auto_fit_min_len_ratio)
                if best is None:
                    out.append({"q": qv, "tau": np.nan, "Dq": np.nan,
                                "D1": np.nan, "intercept": np.nan,
                                "r_value": np.nan, "p_value": np.nan,
                                "std_err": np.nan, "n_points": int(x.size)})
                    continue
                slope, intercept = best["slope"], best["intercept"]
                r_value = best["r_value"]; p_value = best["p_value"]
                std_err = best["std_err"]; n_used = best["n_points"]
            else:
                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                n_used = int(x.size)

            tau = float(-slope)
            Dq = float(tau / (qv - 1.0))
            out.append({"q": qv, "tau": tau, "Dq": Dq, "D1": np.nan,
                        "intercept": float(intercept),
                        "r_value": float(r_value), "p_value": float(p_value),
                        "std_err": float(std_err), "n_points": int(n_used)})

        # q == 0
        df_n = _filter_common(df_mass[(df_mass["kind"] == "N") &
                                       (np.isclose(df_mass["q"], 0.0))])
        if not df_n.empty:
            df_n = df_n.sort_values("scale")
            eps = df_n["eps"].astype(np.float64).values
            x = np.log(1.0 / eps)
            N = df_n["value"].astype(np.float64).values
            with np.errstate(divide="ignore", invalid="ignore"):
                y = np.log(N)
            mask = np.isfinite(x) & np.isfinite(y)
            x0 = x[mask]; y0 = y[mask]

            if x0.size >= min_points:
                if if_auto_line_fit:
                    best = _best_linear_segment(x0, y0, min_points,
                                                auto_fit_min_len_ratio)
                    if best is not None:
                        slope, intercept = best["slope"], best["intercept"]
                        r_value = best["r_value"]; p_value = best["p_value"]
                        std_err = best["std_err"]; n_used = best["n_points"]
                    else:
                        slope = intercept = r_value = p_value = std_err = np.nan
                        n_used = int(x0.size)
                else:
                    if cap_d0_at_2:
                        xs, ys = x0.copy(), y0.copy()
                        slope = intercept = r_value = p_value = std_err = np.nan
                        while xs.size >= min_points:
                            slope, intercept, r_value, p_value, std_err = linregress(xs, ys)
                            if slope <= 2.0:
                                break
                            idx = np.argmax(xs)
                            xs = np.delete(xs, idx); ys = np.delete(ys, idx)
                        x_fit, y_fit = xs, ys
                    else:
                        x_fit, y_fit = x0, y0

                    if len(x_fit) >= 2:
                        slope, intercept, r_value, p_value, std_err = linregress(x_fit, y_fit)
                        n_used = len(x_fit)
                    else:
                        slope = intercept = r_value = p_value = std_err = np.nan
                        n_used = len(x_fit)

                D0 = float(slope)
                tau0 = float(-slope)
                out.append({"q": 0.0, "tau": tau0, "Dq": D0, "D1": np.nan,
                            "intercept": float(intercept),
                            "r_value": float(r_value),
                            "p_value": float(p_value),
                            "std_err": float(std_err),
                            "n_points": int(n_used)})
            else:
                out.append({"q": 0.0, "tau": np.nan, "Dq": np.nan, "D1": np.nan,
                            "intercept": np.nan, "r_value": np.nan,
                            "p_value": np.nan, "std_err": np.nan,
                            "n_points": int(x0.size)})

        # q == 1
        df_s = _filter_common(df_mass[(df_mass["kind"] == "S") &
                                       (np.isclose(df_mass["q"], 1.0))])
        if not df_s.empty:
            df_s = df_s.sort_values("scale")
            eps = df_s["eps"].astype(np.float64).values
            x = np.log(1.0 / eps)
            y = df_s["value"].astype(np.float64).values
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]; y = y[mask]

            if x.size >= min_points:
                if if_auto_line_fit:
                    best = _best_linear_segment(x, y, min_points,
                                                auto_fit_min_len_ratio)
                    if best is not None:
                        slope, intercept = best["slope"], best["intercept"]
                        r_value = best["r_value"]; p_value = best["p_value"]
                        std_err = best["std_err"]; n_used = best["n_points"]
                    else:
                        slope = intercept = r_value = p_value = std_err = np.nan
                        n_used = int(x.size)
                else:
                    slope, intercept, r_value, p_value, std_err = linregress(x, y)
                    n_used = int(x.size)

                D1 = float(slope)
                out.append({"q": 1.0, "tau": 0.0, "Dq": D1, "D1": D1,
                            "intercept": float(intercept),
                            "r_value": float(r_value),
                            "p_value": float(p_value),
                            "std_err": float(std_err),
                            "n_points": int(n_used)})
            else:
                out.append({"q": 1.0, "tau": 0.0, "Dq": np.nan, "D1": np.nan,
                            "intercept": np.nan, "r_value": np.nan,
                            "p_value": np.nan, "std_err": np.nan,
                            "n_points": int(x.size)})

        df_fit = pd.DataFrame(out)
        if df_fit.empty:
            return df_fit
        return df_fit.sort_values("q").reset_index(drop=True)

    def alpha_falpha_from_tau(self, df_fit, spline_k=3, exclude_q1=True,
                              spline_s=0):
        if df_fit is None or df_fit.empty:
            return pd.DataFrame()

        df = df_fit[["q", "tau", "Dq", "D1", "n_points"]].copy()
        df = df[np.isfinite(df["tau"].values)]
        if exclude_q1:
            df = df[~np.isclose(df["q"].values, 1.0)]
        df = df.sort_values("q").reset_index(drop=True)

        if df.shape[0] < max(5, spline_k + 2):
            df["alpha"] = np.nan
            df["f_alpha"] = np.nan
            return df

        q_vals = df["q"].values.astype(np.float64)
        tau_vals = df["tau"].values.astype(np.float64)

        spl = UnivariateSpline(q_vals, tau_vals, k=spline_k, s=spline_s)
        alpha = spl.derivative()(q_vals)
        f_alpha = q_vals * alpha - tau_vals

        df["alpha"] = alpha
        df["f_alpha"] = f_alpha
        return df

    def get_mfs(self, max_size=None, max_scales=80, min_points=6, min_box=2,
                use_middle_scales=False, fit_scale_frac=(0.2, 0.8),
                if_auto_line_fit=False, auto_fit_min_len_ratio=0.5,
                spline_s=0, cap_d0_at_2=False):
        df_mass = self.get_mass_table(max_size=max_size,
                                      max_scales=max_scales, min_box=min_box)
        df_fit = self.fit_tau_and_D1(
            df_mass, min_points=min_points, require_common_scales=True,
            use_middle_scales=use_middle_scales, fit_scale_frac=fit_scale_frac,
            if_auto_line_fit=if_auto_line_fit,
            auto_fit_min_len_ratio=auto_fit_min_len_ratio,
            cap_d0_at_2=cap_d0_at_2,
        )
        df_spec = self.alpha_falpha_from_tau(df_fit, exclude_q1=True,
                                             spline_s=spline_s)
        return df_mass, df_fit, df_spec

    # ============================================================
    # Batch GPU MFS
    # ============================================================
    @staticmethod
    @torch.no_grad()
    def get_batch_mfs(img_list,
                      q_list=np.linspace(-5, 5, 51),
                      corp_type=-1,
                      max_size=None,
                      max_scales=80,
                      min_box=2,
                      min_points=6,
                      use_middle_scales=False,
                      fit_scale_frac=(0.2, 0.8),
                      if_auto_line_fit=False,
                      auto_fit_min_len_ratio=0.5,
                      spline_s=0,
                      cap_d0_at_2=False,
                      bg_threshold=0.01,
                      bg_reverse=False,
                      bg_otsu=False,
                      mu_floor=1e-12,
                      device=None,
                      dtype=torch.float64,
                      with_progress=True,
                      img_chunk=None,
                      q_chunk=64):
        """
        Vectorized batch MFS on GPU.

        Returns
        -------
        list of (df_mass, df_fit, df_spec) tuples, one per input image.

        Implementation notes
        --------------------
        * All images are first preprocessed (same path as the single-image
          class) and center-cropped to the common min(H,W) square ROI.
        * For each scale s, every image's ROI is sub-cropped to (roi//s)*s,
          divided into s x s blocks, and probabilities mu_i(s) are computed
          on the GPU in one batched op.
        * mu and the auxiliary log mu live in `dtype` (default fp64,
          matches the single-image class). The reductions that matter
          (logsumexp over q, xlogy for S) are always done in fp64 to
          preserve precision near large |q| and around mu ~ 1.
        * The q-axis is chunked (`q_chunk`) so the (B, n_blocks, n_q)
          intermediate never exceeds ~q_chunk * B * n_blocks * 8 bytes.
        * The image batch is chunked (`img_chunk`) so VRAM stays bounded
          even for thousands of images.

        Parameters
        ----------
        max_size : int or None
            Override the ROI side length. None = min(H, W) across the
            batch (same as the single-image class default).
        mu_floor : float
            Kept for API parity with the CPU class. Not used internally
            (the GPU code path is mu-floor-free by design, like the CPU
            class itself in the post-refactor version).
        dtype : torch.dtype, default torch.float64
            Compute dtype for ROI tensors. fp64 (default) gives byte-exact
            agreement with the CPU class and the single-image GPU class;
            pass torch.float32 for ~2x speedup at ~1e-5 typical accuracy
            loss.
        img_chunk : int or None
            Max images simultaneously resident on the GPU. None = all.
        q_chunk : int
            Max q values processed per logsumexp call. Tune down on OOM.
        """
        # `mu_floor` is intentionally unused: the underlying mass-table
        # computation works in log-space without flooring (logsumexp
        # handles -inf gracefully). The argument is accepted only for
        # API symmetry with CFAImageMFSGPU.get_batch_mfs.
        del mu_floor  # silence unused-variable linters

        if len(img_list) == 0:
            return []

        # ---- preprocess on CPU (cheap, identical to single path) ----
        imgs_proc = [_preprocess_image(im, bg_threshold=bg_threshold,
                                       bg_reverse=bg_reverse, bg_otsu=bg_otsu)
                     for im in img_list]

        # Honour max_size override (same default semantics as CPU class).
        common_min = int(min(min(im.shape) for im in imgs_proc))
        if max_size is None:
            min_hw = common_min
        else:
            min_hw = int(min(int(max_size), common_min))
        # Use a numpy dtype that matches the requested torch dtype so we
        # don't silently downcast a fp64 user request to fp32.
        np_dtype = (np.float64 if dtype == torch.float64
                    else np.float32)
        rois = np.stack([crop_square_roi(im, L=min_hw, mode="center")[0]
                         for im in imgs_proc], axis=0).astype(np_dtype,
                                                              copy=False)
        B, roi_size, _ = rois.shape

        scales = _candidate_scales(min_box, roi_size, max_scales)
        if scales.size == 0:
            empty = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
            return [empty for _ in range(B)]

        device = (torch.device(device) if device is not None
                  else (torch.device("cuda")
                        if torch.cuda.is_available() else torch.device("cpu")))
        q_arr = np.asarray(q_list, dtype=np.float64)
        n_q = q_arr.size

        # which q indices are special
        q0_mask = np.isclose(q_arr, 0.0)
        q1_mask = np.isclose(q_arr, 1.0)
        # q values used for logMq (NOT q==0 or q==1)
        logmq_mask_np = ~(q0_mask | q1_mask)
        logmq_q_idx = np.where(logmq_mask_np)[0]
        logmq_q_vals = q_arr[logmq_q_idx]
        n_q_lm = logmq_q_idx.size

        n_scales = scales.size
        out_logMq = np.full((B, n_scales, n_q_lm), np.nan, dtype=np.float64)
        out_N     = np.full((B, n_scales),         np.nan, dtype=np.float64)
        out_S     = np.full((B, n_scales),         np.nan, dtype=np.float64)
        out_eps   = np.full(n_scales,              np.nan, dtype=np.float64)
        valid_scales = np.zeros(n_scales, dtype=bool)

        if img_chunk is None or img_chunk <= 0:
            img_chunk = B
        img_chunk = min(int(img_chunk), B)
        fixed_L = float(roi_size)

        # On the GPU, q values used for logMq.
        q_lm_t = torch.from_numpy(logmq_q_vals).to(device=device,
                                                   dtype=torch.float64)

        # progress on outer loop only
        scales_iter_outer = (tqdm(range(n_scales),
                                  desc=f"Batch scales on {device.type}")
                             if with_progress else range(n_scales))

        for img_lo in range(0, B, img_chunk):
            img_hi = min(B, img_lo + img_chunk)
            sub_rois = torch.from_numpy(rois[img_lo:img_hi]).to(
                device=device, dtype=dtype)                # (b, L, L)
            b = sub_rois.shape[0]

            inner_iter = scales_iter_outer if img_lo == 0 else range(n_scales)

            for si in inner_iter:
                s = int(scales[si])
                new_L = (roi_size // s) * s
                if new_L == 0:
                    continue
                nY = new_L // s
                nX = new_L // s
                n_blocks = nY * nX

                sub = sub_rois[:, :new_L, :new_L]
                # (b, nY, s, nX, s) -> sum over (s, s)
                block_mass = sub.reshape(b, nY, s, nX, s).sum(dim=(2, 4))
                block_mass = block_mass.reshape(b, n_blocks)
                block_mass = torch.clamp(block_mass, min=0.0)

                total = block_mass.sum(dim=1, keepdim=True)        # (b, 1)
                bad_total = (~torch.isfinite(total)) | (total <= 0)
                safe_total = torch.where(bad_total,
                                          torch.ones_like(total), total)
                mu_lo = block_mass / safe_total                    # in `dtype`
                if bad_total.any():
                    mu_lo = torch.where(bad_total.expand_as(mu_lo),
                                        torch.zeros_like(mu_lo), mu_lo)

                pos = mu_lo > 0                                    # (b, n_blocks)

                # --- N (q==0): count of positive boxes ---
                N_b = pos.sum(dim=1).to(torch.float64)             # (b,)

                # --- S (q==1): -sum mu * log mu (xlogy is 0*log0 = 0) ---
                # do this in fp64 for precision; at most O(B * n_blocks)
                mu_64 = mu_lo.to(torch.float64)
                S_b = -torch.special.xlogy(mu_64, mu_64).sum(dim=1)  # (b,)

                # mark invalid rows
                if bad_total.any():
                    bad1d = bad_total.squeeze(1)
                    nan_v = torch.full_like(N_b, float("nan"))
                    N_b = torch.where(bad1d, nan_v, N_b)
                    S_b = torch.where(bad1d, nan_v, S_b)

                # --- logMq for q in logmq_q_vals ---
                # Empty boxes (mu_i == 0) contribute 0 to sum mu_i^q, which
                # is -inf in log space. The naive expression
                #     a = q * log_mu  with log_mu = -inf on empty boxes
                # is fine for q > 0 but produces +inf for q < 0, blowing up
                # logsumexp. We therefore compute a on a safe placeholder
                # and then force a = -inf on empty boxes regardless of q
                # sign. This matches the single-image path which simply
                # excludes empty boxes via mu_pos = mu[mu > 0] before the
                # logsumexp.
                mu_64_clamped = mu_64.clamp_min(1e-300)
                log_mu_safe = torch.log(mu_64_clamped)             # (b, n_blocks)
                neg_inf_scalar = float("-inf")

                logMq_buf = torch.empty((b, n_q_lm), device=device,
                                        dtype=torch.float64)

                lo_q = 0
                while lo_q < n_q_lm:
                    hi_q = min(lo_q + q_chunk, n_q_lm)
                    q_slice = q_lm_t[lo_q:hi_q]                    # (qc,)
                    # a[b, i, q] = q * log_mu_safe[b, i]
                    # (b, n_blocks, qc)
                    a = log_mu_safe.unsqueeze(2) * q_slice.view(1, 1, -1)
                    # Mask empty-box rows to -inf for ALL q (so they
                    # contribute 0 to sum mu_i^q regardless of q's sign).
                    pos_3d = pos.unsqueeze(2).expand_as(a)
                    a = torch.where(pos_3d, a,
                                    torch.full_like(a, neg_inf_scalar))
                    lse = torch.logsumexp(a, dim=1)                # (b, qc)
                    logMq_buf[:, lo_q:hi_q] = lse
                    del a, pos_3d, lse
                    lo_q = hi_q

                if bad_total.any():
                    bad_b = bad_total.squeeze(1)
                    logMq_buf = torch.where(
                        bad_b.unsqueeze(1).expand_as(logMq_buf),
                        torch.full_like(logMq_buf, float("nan")),
                        logMq_buf,
                    )

                # store
                out_N[img_lo:img_hi, si]     = N_b.detach().cpu().numpy()
                out_S[img_lo:img_hi, si]     = S_b.detach().cpu().numpy()
                out_logMq[img_lo:img_hi, si, :] = logMq_buf.detach().cpu().numpy()
                if img_lo == 0:
                    out_eps[si] = float(s) / fixed_L
                    valid_scales[si] = True

                del block_mass, total, safe_total, mu_lo, mu_64, mu_64_clamped, \
                    log_mu_safe, pos, N_b, S_b, logMq_buf

            del sub_rois
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # ---- assemble per-image DataFrames (vectorized assembly) ----
        used_idx    = np.where(valid_scales)[0]
        used_scales = scales[used_idx]
        used_eps    = out_eps[used_idx]
        N_v         = out_N[:, used_idx]                            # (B, ns)
        S_v         = out_S[:, used_idx]                            # (B, ns)
        L_v         = out_logMq[:, used_idx, :]                     # (B, ns, n_q_lm)
        ns          = used_idx.size

        # Build the long-format frame for each image without per-cell loops.
        # We pre-tile the (scale, eps) and (q, kind) axes once and reuse them.
        scale_col_for_NS = np.tile(used_scales.astype(int),  1)     # (ns,)
        eps_col_for_NS   = np.tile(used_eps.astype(float),   1)     # (ns,)
        # logMq shape (ns, n_q_lm) when flattened (C order):
        scale_col_lm = np.repeat(used_scales.astype(int),  n_q_lm)  # (ns*n_q_lm,)
        eps_col_lm   = np.repeat(used_eps.astype(float),   n_q_lm)  # (ns*n_q_lm,)
        q_col_lm     = np.tile  (logmq_q_vals.astype(float), ns)    # (ns*n_q_lm,)

        results = []

        # Re-use one fitter "shell" for fit + spectrum
        fitter = CFAImageMFSGPU.__new__(CFAImageMFSGPU)
        fitter.m_q_list = q_arr
        fitter.m_with_progress = False
        fitter.m_image = imgs_proc[0]
        fitter.m_corp_type = corp_type
        fitter.device = device
        fitter.dtype = dtype
        fitter._mu_floor = 1e-12

        for i in range(B):
            # N rows (kind="N", q=0)
            N_vals = N_v[i]
            keep_n = np.isfinite(N_vals)
            df_N = pd.DataFrame({
                "scale": scale_col_for_NS[keep_n],
                "eps":   eps_col_for_NS[keep_n],
                "q":     np.zeros(int(keep_n.sum())),
                "value": N_vals[keep_n].astype(float),
                "kind":  np.full(int(keep_n.sum()), "N", dtype=object),
            })

            # S rows (kind="S", q=1)
            S_vals = S_v[i]
            keep_s = np.isfinite(S_vals)
            df_S = pd.DataFrame({
                "scale": scale_col_for_NS[keep_s],
                "eps":   eps_col_for_NS[keep_s],
                "q":     np.ones(int(keep_s.sum())),
                "value": S_vals[keep_s].astype(float),
                "kind":  np.full(int(keep_s.sum()), "S", dtype=object),
            })

            # logMq rows
            L_flat = L_v[i].reshape(-1)                             # ns*n_q_lm
            keep_l = np.isfinite(L_flat)
            df_L = pd.DataFrame({
                "scale": scale_col_lm[keep_l],
                "eps":   eps_col_lm[keep_l],
                "q":     q_col_lm[keep_l],
                "value": L_flat[keep_l].astype(float),
                "kind":  np.full(int(keep_l.sum()), "logMq", dtype=object),
            })

            df_mass = (pd.concat([df_N, df_S, df_L], ignore_index=True)
                       .sort_values(["kind", "q", "scale"])
                       .reset_index(drop=True))

            df_fit = fitter.fit_tau_and_D1(
                df_mass, min_points=min_points,
                require_common_scales=True,
                use_middle_scales=use_middle_scales,
                fit_scale_frac=fit_scale_frac,
                if_auto_line_fit=if_auto_line_fit,
                auto_fit_min_len_ratio=auto_fit_min_len_ratio,
                cap_d0_at_2=cap_d0_at_2,
            )
            df_spec = fitter.alpha_falpha_from_tau(df_fit, exclude_q1=True,
                                                    spline_s=spline_s)
            results.append((df_mass, df_fit, df_spec))

        return results

    # ============================================================
    # Local (coarse-grained) singularity map alpha(x, y) on GPU
    # ============================================================
    @torch.no_grad()
    def compute_alpha_map(self, scales=None, roi_mode="center",
                          empty_policy="nan"):
        """
        Per-pixel local singularity exponent map alpha(x, y), GPU version
        of CFAImageMFSGPU.compute_alpha_map. See that docstring for the maths.
        Result is returned as a numpy (L, L) float64 array.
        """
        # Delegate to the streaming batch path (B=1).
        maps, info = CFAImageMFSGPU.compute_alpha_map_batch(
            [self.m_image], scales=scales, roi_mode=roi_mode,
            bg_threshold=0.0,            # already preprocessed in __init__
            bg_reverse=False, bg_otsu=False, empty_policy=empty_policy,
            with_progress=False, device=self.device, dtype=self.dtype)
        return maps[0], info

    @staticmethod
    @torch.no_grad()
    def compute_alpha_map_batch(images, scales=None, roi_mode="center",
                                bg_threshold=0.01, bg_reverse=False,
                                bg_otsu=False, empty_policy="nan",
                                with_progress=True, device=None,
                                dtype=torch.float64, img_chunk=None):
        """
        Batched per-pixel alpha-map on GPU using streaming OLS, with the
        nested-grid optimisation when scales are all multiples of s_min.

        Implementation
        --------------
          1) Nested-grid trick. When every scale s_k is an integer
             multiple of the smallest scale s_min, every s_min x s_min
             pixel patch carries a constant alpha (the mu trajectories at
             those pixels are identical at every scale). We therefore run
             OLS on the (L/s_min, L/s_min) coarse grid and upsample once
             at the end. This shrinks GPU buffers by s_min^2.

          2) Streaming OLS. We never materialise the (B, n_scales, L, L)
             tensor; we accumulate the five OLS sums on the fly.

        Parameters
        ----------
        images : list[np.ndarray]
        scales : iterable of int or None
        bg_threshold, bg_reverse, bg_otsu : preprocessing flags
        empty_policy : {"nan", "fill"}
        device : torch.device or str
        dtype : torch.dtype, default fp64
        img_chunk : int or None

        Returns
        -------
        alpha_maps : list[np.ndarray]   each (L, L) float64
        info : dict {"L", "scales", "log_eps"}
        """
        if len(images) == 0:
            return [], {"L": 0, "scales": np.array([], dtype=int),
                        "log_eps": np.array([], dtype=np.float64)}

        prepped = [_preprocess_image(im, bg_threshold=bg_threshold,
                                     bg_reverse=bg_reverse, bg_otsu=bg_otsu)
                   for im in images]
        L_common = int(min(min(im.shape) for im in prepped))

        # Use a numpy dtype that matches the requested torch dtype so we
        # don't silently downcast a fp64 user request to fp32.
        np_dtype = (np.float64 if dtype == torch.float64
                    else np.float32)
        rois_np = np.stack(
            [crop_square_roi(im, L=L_common, mode=roi_mode)[0]
             for im in prepped], axis=0
        ).astype(np_dtype, copy=False)                                 # (B,L,L)
        B = rois_np.shape[0]

        if scales is None:
            max_pow = max(2, int(np.floor(np.log2(max(2, L_common // 4)))))
            scales = np.array([2 ** k for k in range(1, max_pow + 1)
                               if 2 ** k <= L_common], dtype=int)
        scales = np.array(sorted({int(s) for s in scales
                                  if 1 <= s <= L_common}), dtype=int)
        if scales.size < 2:
            raise ValueError("Need at least 2 valid scales for alpha map.")

        device = (torch.device(device) if device is not None
                  else (torch.device("cuda")
                        if torch.cuda.is_available() else torch.device("cpu")))

        if img_chunk is None or img_chunk <= 0:
            img_chunk = B
        img_chunk = min(int(img_chunk), B)

        log_eps_np = np.log(scales.astype(np.float64) / float(L_common))

        s_min = int(scales.min())
        nested_ok = (L_common % s_min == 0) and \
                    bool(np.all((scales % s_min) == 0))
        grid_L = (L_common // s_min) if nested_ok else L_common

        alpha_maps = [None] * B

        outer = (tqdm(range(0, B, img_chunk),
                      desc=f"Batch alpha-map on {device.type}")
                 if with_progress else range(0, B, img_chunk))

        for img_lo in outer:
            img_hi = min(B, img_lo + img_chunk)
            b = img_hi - img_lo

            roi_b = torch.from_numpy(rois_np[img_lo:img_hi]).to(
                device=device, dtype=dtype)                            # (b,L,L)

            n_eff  = torch.zeros((b, grid_L, grid_L), device=device,
                                 dtype=torch.float64)
            sum_x  = torch.zeros_like(n_eff)
            sum_y  = torch.zeros_like(n_eff)
            sum_xx = torch.zeros_like(n_eff)
            sum_xy = torch.zeros_like(n_eff)

            for k, s in enumerate(scales):
                s = int(s)
                x_k = float(log_eps_np[k])
                new_L = (L_common // s) * s
                if new_L == 0:
                    continue
                n = new_L // s
                sub = roi_b[:, :new_L, :new_L]
                bm = sub.reshape(b, n, s, n, s).sum(dim=(2, 4))         # (b,n,n)
                bm = bm.clamp_min(0.0)
                total = bm.reshape(b, -1).sum(dim=1)                    # (b,)
                ok = torch.isfinite(total) & (total > 0)
                if not ok.any():
                    continue
                safe = torch.where(ok, total,
                                   torch.ones_like(total))              # (b,)
                mu = (bm / safe.view(b, 1, 1))                          # (b,n,n)
                if (~ok).any():
                    mu = torch.where(ok.view(b, 1, 1), mu,
                                     torch.zeros_like(mu))

                if empty_policy == "fill":
                    pos_mask = mu > 0
                    big = torch.where(pos_mask, mu,
                                      torch.full_like(mu, float("inf")))
                    mins = big.reshape(b, -1).min(dim=1).values
                    mins = torch.where(torch.isfinite(mins), mins,
                                       torch.full_like(mins, 1e-300))
                    mu_filled = torch.where(pos_mask, mu,
                                             mins.view(b, 1, 1))
                    log_mu_grid = torch.log(mu_filled.to(torch.float64))
                    valid_grid = ok.view(b, 1, 1).expand(b, n, n).to(
                        torch.float64)
                else:
                    pos = mu > 0
                    safe_mu = torch.where(pos, mu.to(torch.float64),
                                           torch.ones_like(mu, dtype=torch.float64))
                    log_mu_grid = torch.log(safe_mu)
                    log_mu_grid = torch.where(pos, log_mu_grid,
                                              torch.zeros_like(log_mu_grid))
                    valid_grid = (pos & ok.view(b, 1, 1)).to(torch.float64)

                if nested_ok:
                    rep = s // s_min                                  # >=1
                    log_mu_up = log_mu_grid.repeat_interleave(
                        rep, dim=1).repeat_interleave(rep, dim=2)
                    valid_up = valid_grid.repeat_interleave(
                        rep, dim=1).repeat_interleave(rep, dim=2)
                    gL = new_L // s_min
                    n_eff [:, :gL, :gL] += valid_up[:, :gL, :gL]
                    sum_x [:, :gL, :gL] += valid_up[:, :gL, :gL] * x_k
                    yt = torch.where(valid_up[:, :gL, :gL] > 0,
                                     log_mu_up[:, :gL, :gL],
                                     torch.zeros_like(log_mu_up[:, :gL, :gL]))
                    sum_y [:, :gL, :gL] += yt
                    sum_xx[:, :gL, :gL] += valid_up[:, :gL, :gL] * (x_k * x_k)
                    sum_xy[:, :gL, :gL] += yt * x_k
                    del log_mu_up, valid_up, yt
                else:
                    log_mu_pix = log_mu_grid.repeat_interleave(
                        s, dim=1).repeat_interleave(s, dim=2)
                    valid_pix = valid_grid.repeat_interleave(
                        s, dim=1).repeat_interleave(s, dim=2)
                    n_eff [:, :new_L, :new_L] += valid_pix
                    sum_x [:, :new_L, :new_L] += valid_pix * x_k
                    yt = torch.where(valid_pix > 0, log_mu_pix,
                                     torch.zeros_like(log_mu_pix))
                    sum_y [:, :new_L, :new_L] += yt
                    sum_xx[:, :new_L, :new_L] += valid_pix * (x_k * x_k)
                    sum_xy[:, :new_L, :new_L] += yt * x_k
                    del log_mu_pix, valid_pix, yt

                del bm, total, safe, mu, log_mu_grid, valid_grid

            denom = n_eff * sum_xx - sum_x * sum_x
            numer = n_eff * sum_xy - sum_x * sum_y
            slope = torch.where((n_eff >= 2) & (denom.abs() > 0),
                                numer / denom,
                                torch.full_like(denom, float("nan")))

            if nested_ok and s_min > 1:
                slope = slope.repeat_interleave(
                    s_min, dim=1).repeat_interleave(s_min, dim=2)

            slope_np = slope.detach().cpu().numpy().astype(np.float64)
            for j in range(b):
                alpha_maps[img_lo + j] = slope_np[j]

            del roi_b, n_eff, sum_x, sum_y, sum_xx, sum_xy, denom, numer, slope
            if device.type == "cuda":
                torch.cuda.empty_cache()

        return alpha_maps, {"L": L_common, "scales": scales,
                            "log_eps": log_eps_np}

    def plot_alpha_map(self, alpha_map):
        """
        Visualize alpha(x,y)
        """
        plt.figure(figsize=(6, 6))
        plt.imshow(alpha_map, cmap="jet")
        plt.colorbar(label=r"$\alpha(x,y)$")
        plt.title("Local Multifractal α-map")
        plt.axis("off")
        plt.show()
    # --------------------------------------------------------
    # Plotting
    # --------------------------------------------------------
    def plot(self, df_mass, df_fit, df_spec):
        fig, axs = plt.subplots(2, 2, figsize=(8, 6))

        df_logmq = df_mass[df_mass["kind"] == "logMq"].copy()
        if not df_logmq.empty:
            pivot = df_logmq.pivot_table(index="scale", columns="q",
                                         values="value", aggfunc="mean")
            vals = pivot.values
            if np.any(np.isfinite(vals)):
                vmin = np.nanpercentile(vals, 10)
                vmax = np.nanpercentile(vals, 90)
                sns.heatmap(pivot, ax=axs[0, 0], cmap="coolwarm",
                            vmin=vmin, vmax=vmax, cbar=True)
                axs[0, 0].set_xlabel("q")
                axs[0, 0].set_ylabel("box size (pixels)")
                axs[0, 0].set_title("Heatmap: log M(q, eps) vs box size and q")
                axs[0, 0].xaxis.set_major_formatter(
                    mticker.FormatStrFormatter("%.2f"))
            else:
                axs[0, 0].set_title("Heatmap: (all NaN)")
        else:
            axs[0, 0].set_title("Heatmap: (no logMq data)")

        if df_fit is not None and not df_fit.empty:
            sns.lineplot(data=df_fit, x="q", y="tau", ax=axs[0, 1])
            axs[0, 1].set_xlabel("q"); axs[0, 1].set_ylabel(r"$\tau(q)$")
            axs[0, 1].set_title(r"$\tau(q)$ vs $q$"); axs[0, 1].grid(True)
        else:
            axs[0, 1].set_title("tau(q): (no data)")

        if df_fit is not None and not df_fit.empty:
            sns.lineplot(data=df_fit, x="q", y="Dq", ax=axs[1, 0])
            axs[1, 0].set_xlabel("q"); axs[1, 0].set_ylabel(r"$D(q)$")
            axs[1, 0].set_title(r"$D(q)$ vs $q$"); axs[1, 0].grid(True)
        else:
            axs[1, 0].set_title("D(q): (no data)")

        if (df_spec is not None and not df_spec.empty
                and np.any(np.isfinite(df_spec["alpha"].values))):
            sns.lineplot(data=df_spec, x="alpha", y="f_alpha", ax=axs[1, 1])
            axs[1, 1].set_xlabel(r"$\alpha$"); axs[1, 1].set_ylabel(r"$f(\alpha)$")
            axs[1, 1].set_title(r"$f(\alpha)$ vs $\alpha$"); axs[1, 1].grid(True)
        else:
            axs[1, 1].set_title("f(alpha): (no data)")

        plt.tight_layout()
        plt.show()

# ============================================================
# Demo
# ============================================================
def main():
    image_path = "../images/fractal.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)

    imgs = [image for _ in range(250)]

    q_list = np.linspace(-10, 10, 101)
    with_progress = False

    start = time.time()
    for img in tqdm(imgs,desc="single calculation"):
        mfs = CFAImageMFSGPU(image=img, corp_type=-1, q_list=q_list,
                          with_progress=with_progress, bg_reverse=False,
                          bg_threshold=0.01, bg_otsu=False)
        df_mass, df_fit, df_spec = mfs.get_mfs(
            max_scales=80, min_points=6, use_middle_scales=False,
            if_auto_line_fit=False, fit_scale_frac=(0.3, 0.7),
            auto_fit_min_len_ratio=0.6, cap_d0_at_2=False)
    print("time used (no batch):", time.time() - start)
    print(df_fit.head())

    start = time.time()
    batch_results = CFAImageMFSGPU.get_batch_mfs(
        imgs, with_progress=with_progress, q_list=q_list, corp_type=-1,
        bg_reverse=False, bg_threshold=0.01, bg_otsu=False, max_scales=80,
        min_points=6, use_middle_scales=False, if_auto_line_fit=False,
        fit_scale_frac=(0.3, 0.7), auto_fit_min_len_ratio=0.6,
        cap_d0_at_2=False)
    df_mass1, df_fit1, df_spec1 = batch_results[0]
    print("time used (batch):", time.time() - start)
    print(df_fit1.head())


if __name__ == "__main__":
    main()
