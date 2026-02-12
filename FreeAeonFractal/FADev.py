#bata代码，优化q<0的处理逻辑
#原逻辑（常见）： q < 0 q<0 时 不让  μ = 0 μ=0 参与  Z ( q ) Z(q)（只对  μ > 0 μ>0 求和）。 
#你现在代码： q < 0 q<0 时把  μ = 0 μ=0 变成 mu_floor 强行参与，导致负 q 端对空盒极其敏感，结果可能被显著扭曲
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from scipy.stats import linregress
from scipy.interpolate import UnivariateSpline
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
from skimage.util import view_as_blocks


# ============================================================
# Fixed ROI utilities (Scheme A: fixed square ROI, no LCM)
# ============================================================
def crop_square_roi(img, L=None, mode="center"):
    """
    Crop a 2D image to a fixed square ROI of size LxL.
    If L is None: use L = min(H, W).

    mode:
      - "topleft": crop from top-left corner
      - "center":  centered crop
    """
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
# Image blocking utilities (2D only)
# ============================================================
class CFAImage:
    @staticmethod
    def crop_data(data, block_size):
        """Crop spatial dimensions so H and W are multiples of block_size."""
        if data is None:
            raise ValueError("data is None")
        bh, bw = block_size
        h, w = data.shape[:2]
        new_h = (h // bh) * bh
        new_w = (w // bw) * bw
        return data[:new_h, :new_w]

    @staticmethod
    def get_boxes_from_image(image, block_size, corp_type=-1):
        """
        Split a 2D grayscale image into blocks.

        Returns:
            blocks_reshaped: (num_blocks, bh, bw)
            raw_blocks:      (nY, nX, bh, bw)
        """
        if image.ndim != 2:
            raise ValueError("For MFS, image must be 2D grayscale (H,W).")

        bh, bw = block_size

        if corp_type == -1:
            img = CFAImage.crop_data(image, block_size)
        elif corp_type == 0:
            # strict: require exact tiling, otherwise return empty
            if image.shape[0] % bh != 0 or image.shape[1] % bw != 0:
                return (np.empty((0, bh, bw), dtype=image.dtype),
                        np.empty((0, 0, bh, bw), dtype=image.dtype))
            img = image
        else:
            img = CFAImage.crop_data(image, block_size)

        if img.shape[0] < bh or img.shape[1] < bw:
            return (np.empty((0, bh, bw), dtype=img.dtype),
                    np.empty((0, 0, bh, bw), dtype=img.dtype))

        raw_blocks = view_as_blocks(img, block_shape=(bh, bw))  # (nY, nX, bh, bw)
        nY, nX = raw_blocks.shape[:2]
        blocks_reshaped = raw_blocks.reshape(nY * nX, bh, bw)
        return blocks_reshaped, raw_blocks


# ============================================================
# Box-counting multifractal spectrum (MFS)
# ============================================================
class CFA2DMFSBoxCounting:
    """
    Box-counting multifractal analysis on a grayscale image.

    Parameters
    ----------
    image : 2D array
        Input grayscale image.
    corp_type : int
        -1 crop to multiples of box size (OLD; will break fixed ROI idea).
         0 require exact multiples (recommended with Scheme A).
    q_list : array-like
        q values. Can include 1; we compute D1 specially.
    with_progress : bool
        Show progress bars.
    bg_threshold : float, default 1e-6
        After normalization, pixels < bg_threshold are set to 0.
    bg_otsu : bool, default True
        Apply Otsu threshold on raw image to remove background.
    mu_floor : float, default 1e-12
        Used in Scheme A fix: floor μ==0 before log for q!=0,1 (especially negative q).
    """

    def __init__(self, image, corp_type=-1, q_list=np.linspace(-5, 5, 51),
                 with_progress=True, bg_threshold=1e-6, bg_reverse=False, bg_otsu=True, mu_floor=1e-12):
        if image is None:
            raise ValueError("image is None")
        if image.ndim != 2:
            raise ValueError("image must be a 2D grayscale array (H,W).")

        img_raw = image.astype(np.float64)

        # optional Otsu on raw image
        if bg_otsu:
            vmin = np.nanmin(img_raw)
            vmax = np.nanmax(img_raw)
            if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                img8 = ((img_raw - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
                _, img_bin = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                img_raw[img_bin == 0] = 0.0

        # Normalize to [0,1] and ensure nonnegative measure
        img = img_raw.copy()
        img -= np.nanmin(img)
        mx = np.nanmax(img)
        img /= (mx + 1e-12)
        img = np.where(np.isfinite(img), img, 0.0)
        img = np.clip(img, 0.0, 1.0)

        # background thresholding (post-normalization)
        self._bg_threshold = bg_threshold
        if bg_threshold > 0:
            if bg_reverse:
                img[img > bg_threshold] = 0.0
            else:
                img[img < bg_threshold] = 0.0

        self.m_image = img
        self.m_corp_type = corp_type
        self.m_with_progress = with_progress
        self.m_q_list = np.array(q_list, dtype=np.float64)

        self._mu_floor = float(mu_floor)
        if not np.isfinite(self._mu_floor) or self._mu_floor <= 0:
            raise ValueError("mu_floor must be finite and > 0")

    # ------------------------------------------------------------
    # Core: compute per-scale measures μ_i(ε)
    # ------------------------------------------------------------
    def _mu_at_scale(self, box_size):
        """
        Compute box probabilities μ_i at a given box size.

        Returns:
            mu : 1D float64 array, μ_i >= 0, sum(mu)=1
        """
        boxes, _ = CFAImage.get_boxes_from_image(
            self.m_image, (box_size, box_size), corp_type=self.m_corp_type
        )
        if boxes.size == 0:
            return np.array([], dtype=np.float64)

        block_mass = np.sum(boxes, axis=(1, 2)).astype(np.float64)
        block_mass = np.where(block_mass < 0, 0.0, block_mass)

        total = float(np.sum(block_mass))
        if (not np.isfinite(total)) or total <= 0:
            return np.array([], dtype=np.float64)

        mu = block_mass / total
        mu = np.where(np.isfinite(mu) & (mu >= 0), mu, 0.0)

        s = float(np.sum(mu))
        if s <= 0 or (not np.isfinite(s)):
            return np.array([], dtype=np.float64)
        mu /= s
        return mu

    # ------------------------------------------------------------
    # Compute per-scale values: logMq (q!=0,1), N (q=0), S (q=1)
    # ------------------------------------------------------------
    def get_mass_table(self, max_size=None, max_scales=80, min_box=2, roi_mode="center"):
        """
        Output columns: scale, eps, q, value, kind

        kind meanings:
            - "logMq": for q != 0,1; value = log( sum_i mu_i^q )
            - "N":     for q == 0;   value = N_nonzero (count of mu_i>0)
            - "S":     for q == 1;   value = -sum_i mu_i log mu_i
        """
        img0 = self.m_image
        h0, w0 = img0.shape
        if max_size is None:
            max_size = min(h0, w0)
        if max_size < min_box:
            raise ValueError("max_size too small.")

        # -------- Scheme A: fixed square ROI once --------
        img_fixed, fixed_L_int = crop_square_roi(img0, L=min(min(h0, w0), max_size), mode=roi_mode)
        fixed_L = float(fixed_L_int)

        # Candidate integer box sizes
        scales = np.logspace(np.log2(min_box), np.log2(fixed_L_int), num=max_scales, base=2.0)
        scales = np.unique(np.maximum(min_box, np.round(scales).astype(int)))
        scales = scales[(scales >= min_box) & (scales <= fixed_L_int)]

        # keep only divisors of fixed_L => exact tiling, no per-scale crop
        scales = scales[(fixed_L_int % scales) == 0]
        scales = np.array(sorted(scales.astype(int)))

        if scales.size == 0:
            return pd.DataFrame()

        # Temporarily replace image with fixed ROI
        old_img = self.m_image
        self.m_image = img_fixed

        records = []
        iterator = tqdm(scales, desc="Computing per-scale μ") if self.m_with_progress else scales

        try:
            for size in iterator:
                size = int(size)
                if size < min_box:
                    continue

                mu = self._mu_at_scale(size)
                if mu.size == 0:
                    continue

                eps = float(size) / fixed_L

                # For q=0 and q=1 we still need mu>0 subset
                mu_pos = mu[mu > 0]
                if mu_pos.size == 0:
                    continue
                log_mu_pos = np.log(mu_pos)

                # For q!=0,1 we use ALL boxes, with μ==0 floored
                mu_all = mu.copy()
                mu_all[mu_all == 0] = self._mu_floor
                # guard (shouldn't happen, but avoid log(neg/NaN))
                mu_all = np.where(np.isfinite(mu_all) & (mu_all > 0), mu_all, self._mu_floor)
                log_mu_all = np.log(mu_all)

                for q in self.m_q_list:
                    q = float(q)

                    if np.isclose(q, 1.0):
                        # S = -sum_{mu>0} mu log mu
                        S = float(-np.sum(mu_pos * log_mu_pos))
                        if np.isfinite(S):
                            records.append({"scale": size, "eps": eps, "q": 1.0, "value": S, "kind": "S"})
                        continue

                    if np.isclose(q, 0.0):
                        # N = number of non-empty boxes
                        records.append({"scale": size, "eps": eps, "q": 0.0, "value": float(mu_pos.size), "kind": "N"})
                        continue

                    # log M(q,eps) = log sum_i mu_i^q, computed in log-space
                    a = q * log_mu_all
                    a = a[np.isfinite(a)]
                    logMq = float(logsumexp(a)) if a.size else np.nan
                    if np.isfinite(logMq):
                        records.append({"scale": size, "eps": eps, "q": q, "value": logMq, "kind": "logMq"})
        finally:
            self.m_image = old_img

        df = pd.DataFrame(records)
        if df.empty:
            return df
        return df.sort_values(["kind", "q", "scale"]).reset_index(drop=True)

    # ------------------------------------------------------------
    # Helper: choose a common scale set for regression
    # ------------------------------------------------------------
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
            df_s = df_mass[(df_mass["kind"] == "S") & (np.isclose(df_mass["q"], 1.0))]
            if df_s.empty:
                return np.array([], dtype=int)
            common = common & set(df_s["scale"].astype(int).tolist())

        return np.array(sorted(common), dtype=int)

    # ------------------------------------------------------------
    # Fit τ(q) and D1 (x=log(1/ε))
    # ------------------------------------------------------------
    def fit_tau_and_D1(
        self, df_mass, min_points=6, require_common_scales=True,
        use_middle_scales=True, fit_scale_frac=(0.2, 0.8),
        if_auto_line_fit=False, auto_fit_min_len_ratio=0.5,
        cap_d0_at_2=False
    ):
        if df_mass is None or df_mass.empty:
            return pd.DataFrame()

        df_mass = df_mass.copy()
        df_mass["scale"] = df_mass["scale"].astype(int)

        require_q1 = np.any(np.isclose(self.m_q_list, 1.0))
        if require_common_scales:
            common_scales = self._common_scales_for_fit(df_mass, require_q1=require_q1)
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
            if use_middle and middle_scales is not None and len(middle_scales) > 0:
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
                    xs = x[i:j]
                    ys = y[i:j]
                    slope, intercept, r_value, p_value, std_err = linregress(xs, ys)
                    r2 = r_value ** 2
                    length = j - i
                    cand = dict(slope=slope, intercept=intercept,
                                r_value=r_value, p_value=p_value, std_err=std_err,
                                r2=r2, n_points=length)
                    if best is None:
                        best = cand
                    else:
                        if (length > best["n_points"]) or (length == best["n_points"] and r2 > best["r2"]):
                            best = cand
            return best

        out = []

        # --- q != 0,1: y = logMq ---
        df_logmq = _filter_common(df_mass[df_mass["kind"] == "logMq"])
        for q, df_q in df_logmq.groupby("q"):
            qv = float(q)
            if np.isclose(qv, 1.0) or np.isclose(qv, 0.0):
                continue

            df_q = df_q.sort_values("scale")
            eps = df_q["eps"].astype(np.float64).values
            x = np.log(1.0 / eps)
            y = df_q["value"].astype(np.float64).values  # already logMq

            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]

            if x.size < min_points:
                out.append({"q": qv, "tau": np.nan, "Dq": np.nan, "D1": np.nan,
                            "intercept": np.nan, "r_value": np.nan, "p_value": np.nan,
                            "std_err": np.nan, "n_points": int(x.size)})
                continue

            if if_auto_line_fit:
                best = _best_linear_segment(x, y, min_points, auto_fit_min_len_ratio)
                if best is None:
                    out.append({"q": qv, "tau": np.nan, "Dq": np.nan, "D1": np.nan,
                                "intercept": np.nan, "r_value": np.nan, "p_value": np.nan,
                                "std_err": np.nan, "n_points": int(x.size)})
                    continue
                slope = best["slope"]
                intercept = best["intercept"]
                r_value = best["r_value"]
                p_value = best["p_value"]
                std_err = best["std_err"]
                n_used = best["n_points"]
            else:
                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                n_used = int(x.size)

            # x = log(1/eps) = -log(eps); log M ~ -tau * x  => tau = -slope
            tau = float(-slope)
            Dq = float(tau / (qv - 1.0))
            out.append({"q": qv, "tau": tau, "Dq": Dq, "D1": np.nan,
                        "intercept": float(intercept), "r_value": float(r_value),
                        "p_value": float(p_value), "std_err": float(std_err),
                        "n_points": int(n_used)})

        # --- q == 0: y = log N ---
        df_n = _filter_common(df_mass[(df_mass["kind"] == "N") & (np.isclose(df_mass["q"], 0.0))])
        if not df_n.empty:
            df_n = df_n.sort_values("scale")
            eps = df_n["eps"].astype(np.float64).values
            x = np.log(1.0 / eps)
            N = df_n["value"].astype(np.float64).values
            y = np.log(N)

            mask = np.isfinite(x) & np.isfinite(y)
            x0 = x[mask]
            y0 = y[mask]

            if x0.size >= min_points:
                if if_auto_line_fit:
                    best = _best_linear_segment(x0, y0, min_points, auto_fit_min_len_ratio)
                    if best is not None:
                        slope = best["slope"]
                        intercept = best["intercept"]
                        r_value = best["r_value"]
                        p_value = best["p_value"]
                        std_err = best["std_err"]
                        n_used = best["n_points"]
                    else:
                        slope = intercept = r_value = p_value = std_err = np.nan
                        n_used = int(x0.size)
                else:
                    if cap_d0_at_2:
                        xs, ys = x0.copy(), y0.copy()
                        while xs.size >= min_points:
                            slope, intercept, r_value, p_value, std_err = linregress(xs, ys)
                            if slope <= 2.0:
                                break
                            idx = np.argmax(xs)  # largest x = smallest eps
                            xs = np.delete(xs, idx)
                            ys = np.delete(ys, idx)
                        x_fit, y_fit = xs, ys
                    else:
                        x_fit, y_fit = x0, y0

                    if len(x_fit) >= 2:
                        slope, intercept, r_value, p_value, std_err = linregress(x_fit, y_fit)
                        n_used = len(x_fit)
                    else:
                        slope = intercept = r_value = p_value = std_err = np.nan
                        n_used = len(x_fit)

                tau0 = float(-slope)   # τ(0) = -D0
                D0 = float(slope)
                out.append({"q": 0.0, "tau": tau0, "Dq": D0, "D1": np.nan,
                            "intercept": float(intercept), "r_value": float(r_value),
                            "p_value": float(p_value), "std_err": float(std_err),
                            "n_points": int(n_used)})
            else:
                out.append({"q": 0.0, "tau": np.nan, "Dq": np.nan, "D1": np.nan,
                            "intercept": np.nan, "r_value": np.nan, "p_value": np.nan,
                            "std_err": np.nan, "n_points": int(x0.size)})

        # --- q == 1: fit S vs log(1/eps) ---
        df_s = _filter_common(df_mass[(df_mass["kind"] == "S") & (np.isclose(df_mass["q"], 1.0))])
        if not df_s.empty:
            df_s = df_s.sort_values("scale")
            eps = df_s["eps"].astype(np.float64).values
            x = np.log(1.0 / eps)
            y = df_s["value"].astype(np.float64).values

            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]

            if x.size >= min_points:
                if if_auto_line_fit:
                    best = _best_linear_segment(x, y, min_points, auto_fit_min_len_ratio)
                    if best is not None:
                        slope = best["slope"]
                        intercept = best["intercept"]
                        r_value = best["r_value"]
                        p_value = best["p_value"]
                        std_err = best["std_err"]
                        n_used = best["n_points"]
                    else:
                        slope = intercept = r_value = p_value = std_err = np.nan
                        n_used = int(x.size)
                else:
                    slope, intercept, r_value, p_value, std_err = linregress(x, y)
                    n_used = int(x.size)

                D1 = float(slope)  # S ~ D1 * log(1/eps)
                out.append({"q": 1.0, "tau": 0.0, "Dq": D1, "D1": D1,
                            "intercept": float(intercept), "r_value": float(r_value),
                            "p_value": float(p_value), "std_err": float(std_err),
                            "n_points": int(n_used)})
            else:
                out.append({"q": 1.0, "tau": 0.0, "Dq": np.nan, "D1": np.nan,
                            "intercept": np.nan, "r_value": np.nan, "p_value": np.nan,
                            "std_err": np.nan, "n_points": int(x.size)})

        df_fit = pd.DataFrame(out)
        if df_fit.empty:
            return df_fit
        return df_fit.sort_values("q").reset_index(drop=True)

    # ------------------------------------------------------------
    # α(q) and f(α) from τ(q)
    # ------------------------------------------------------------
    def alpha_falpha_from_tau(self, df_fit, spline_k=3, exclude_q1=True, spline_s=0):
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

    # ------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------
    def get_mfs(self, max_size=None, max_scales=80, min_points=6, min_box=2,
                use_middle_scales=True, fit_scale_frac=(0.2, 0.8),
                if_auto_line_fit=False, auto_fit_min_len_ratio=0.5,
                spline_s=0, cap_d0_at_2=True):
        df_mass = self.get_mass_table(max_size=max_size, max_scales=max_scales, min_box=min_box)
        df_fit = self.fit_tau_and_D1(
            df_mass, min_points=min_points, require_common_scales=True,
            use_middle_scales=use_middle_scales, fit_scale_frac=fit_scale_frac,
            if_auto_line_fit=if_auto_line_fit,
            auto_fit_min_len_ratio=auto_fit_min_len_ratio,
            cap_d0_at_2=cap_d0_at_2
        )
        df_spec = self.alpha_falpha_from_tau(df_fit, exclude_q1=True, spline_s=spline_s)
        return df_mass, df_fit, df_spec

    # ------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------
    def plot(self, df_mass, df_fit, df_spec):
        fig, axs = plt.subplots(2, 2, figsize=(8, 6))

        # Heatmap: logMq for q!=0,1
        df_logmq = df_mass[df_mass["kind"] == "logMq"].copy()
        if not df_logmq.empty:
            pivot = df_logmq.pivot_table(index="scale", columns="q", values="value", aggfunc="mean")
            vals = pivot.values
            if np.any(np.isfinite(vals)):
                vmin = np.nanpercentile(vals, 10)
                vmax = np.nanpercentile(vals, 90)
                sns.heatmap(pivot, ax=axs[0, 0], cmap="coolwarm", vmin=vmin, vmax=vmax, cbar=True)
                axs[0, 0].set_xlabel("q")
                axs[0, 0].set_ylabel("box size (pixels)")
                axs[0, 0].set_title("Heatmap: log M(q, ε) vs box size and q")
                axs[0, 0].xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            else:
                axs[0, 0].set_title("Heatmap: (all NaN)")
        else:
            axs[0, 0].set_title("Heatmap: (no logMq data)")

        if df_fit is not None and not df_fit.empty:
            sns.lineplot(data=df_fit, x="q", y="tau", ax=axs[0, 1])
            axs[0, 1].set_xlabel("q")
            axs[0, 1].set_ylabel(r"$\tau(q)$")
            axs[0, 1].set_title(r"$\tau(q)$ vs $q$")
            axs[0, 1].grid(True)
        else:
            axs[0, 1].set_title("tau(q): (no data)")

        if df_fit is not None and not df_fit.empty:
            sns.lineplot(data=df_fit, x="q", y="Dq", ax=axs[1, 0])
            axs[1, 0].set_xlabel("q")
            axs[1, 0].set_ylabel(r"$D(q)$")
            axs[1, 0].set_title(r"$D(q)$ vs $q$ (with $D_1$ at q=1)")
            axs[1, 0].grid(True)
        else:
            axs[1, 0].set_title("D(q): (no data)")

        if df_spec is not None and not df_spec.empty and np.any(np.isfinite(df_spec["alpha"].values)):
            sns.lineplot(data=df_spec, x="alpha", y="f_alpha", ax=axs[1, 1])
            axs[1, 1].set_xlabel(r"$\alpha$")
            axs[1, 1].set_ylabel(r"$f(\alpha)$")
            axs[1, 1].set_title(r"Multifractal spectrum: $f(\alpha)$ vs $\alpha$")
            axs[1, 1].grid(True)
        else:
            axs[1, 1].set_title("f(alpha): (no data)")

        plt.tight_layout()
        plt.show()


# ============================================================
# Demo
# ============================================================
def main():
    image_path = "../images/test.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)

    q_list = np.linspace(-5, 5, 51)

    mfs = CFA2DMFSBoxCounting(
        image=image,
        corp_type=0,          # Scheme A 下建议用 0：禁止每尺度 crop
        q_list=q_list,
        with_progress=True,
        bg_reverse=True,
        bg_threshold=0.5,
        bg_otsu=False,
        mu_floor=1e-12        # 方案A：用于 μ==0 的 floor
    )

    print("Nonzero pixel ratio (adjust bg_threshold, typically < 0.5):", np.mean(mfs.m_image > 0))

    df_mass, df_fit, df_spec = mfs.get_mfs(
        max_scales=80,
        min_points=6,
        use_middle_scales=False,
        if_auto_line_fit=False,
        fit_scale_frac=(0.3, 0.7),
        auto_fit_min_len_ratio=0.6,
        cap_d0_at_2=False
    )

    if df_fit.empty:
        print("empty fit")
        return

    bad = df_fit[np.isfinite(df_fit["Dq"]) & (df_fit["Dq"] > 2.0)]
    print(bad[["q", "Dq", "tau", "n_points", "r_value"]].head(20))

    print("\n=== Fit results (head) ===")
    print(df_fit.head(10))

    row_d1 = df_fit[np.isclose(df_fit["q"], 1.0)]
    if not row_d1.empty:
        print("\n=== D1 (information dimension) ===")
        print(row_d1[["q", "D1", "Dq", "n_points", "r_value", "std_err"]])

    row_d0 = df_fit[np.isclose(df_fit["q"], 0.0)]
    if not row_d0.empty:
        print("\n=== D0 (capacity dimension) ===")
        print(row_d0[["q", "Dq", "n_points", "r_value", "std_err"]])
        print(f"  D0 should be ≤ 2.0; measured = {row_d0['Dq'].values[0]:.4f}")

    mfs.plot(df_mass, df_fit, df_spec)


if __name__ == "__main__":
    main()

