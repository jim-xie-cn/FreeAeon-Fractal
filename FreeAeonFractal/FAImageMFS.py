"""
Improved Multifractal (box-counting) analysis for 2D grayscale images.

Applied modifications (important):
1) For q != 0,1: compute log M(q, eps) in log-space via logsumexp WITHOUT mu_floor
   - store logMq directly (no exp, no overflow-based dropping)
2) For q=0: store N (count of nonzero boxes), fit with log(N)
3) For q=1: store S = -sum(mu log mu), fit S vs log(1/eps)
4) Keep epsilon normalization with fixed ROI across scales (FIXED ROI; Scheme A):
   - crop ONCE to a fixed square ROI
   - restrict scales to divisors of ROI size => no scale-dependent cropping, no LCM explosion

Notes:
- mu_floor parameter is kept for backward compatibility but is not used anymore (by design).
"""

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
try:
    from FreeAeonFractal.FAImage import CFAImage
except Exception:  # pragma: no cover - fallback for direct import
    from FAImage import CFAImage

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
# Box-counting multifractal spectrum (MFS)
# ============================================================
class CFAImageMFS:
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
        Kept for compatibility; NOT used anymore (we do not floor mu).
    """
    def __init__(self, image, corp_type=-1, q_list=np.linspace(-5, 5, 51),
                 with_progress=True, bg_threshold=0.01, bg_reverse=False, bg_otsu=False, mu_floor=1e-12):
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

        # kept (not used for mu flooring now)
        self._mu_floor = mu_floor

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
        # fixed_L is the ROI side length used for eps normalization (constant across scales)
        img_fixed, fixed_L_int = crop_square_roi(img0, L=min(min(h0, w0), max_size), mode=roi_mode)
        fixed_L = float(fixed_L_int)

        # Candidate integer box sizes (same as before, but will be filtered)
        scales = np.logspace(np.log2(min_box), np.log2(fixed_L_int), num=max_scales, base=2.0)
        scales = np.unique(np.maximum(min_box, np.round(scales).astype(int)))
        scales = scales[(scales >= min_box) & (scales <= fixed_L_int)]

        # keep only divisors of fixed_L => exact tiling, no per-scale crop
        # scales = scales[(fixed_L_int % scales) == 0]
        scales = np.array(sorted(scales.astype(int)))
        if scales.size == 0:
            return pd.DataFrame()

        # Temporarily replace image with fixed ROI (avoid changing other code)
        old_img = self.m_image
        self.m_image = img_fixed
        # ------------------------------------------------

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

                mu_pos_all = mu[mu > 0]
                if mu_pos_all.size == 0:
                    continue

                log_mu_all = np.log(mu_pos_all)
                eps = float(size) / fixed_L

                # Always record N (for D0) and S (for D1) regardless of
                # whether q=0 or q=1 is in q_list. D0 (capacity) and D1
                # (information dimension) are the standard quantities a
                # multifractal analysis is expected to produce, and the
                # GPU class always emits these rows -- doing the same on
                # CPU keeps the two implementations API-compatible.
                S_val = float(-np.sum(mu_pos_all * log_mu_all))
                if np.isfinite(S_val):
                    records.append({"scale": size, "eps": eps, "q": 1.0,
                                    "value": S_val, "kind": "S"})
                records.append({"scale": size, "eps": eps, "q": 0.0,
                                "value": float(mu_pos_all.size),
                                "kind": "N"})

                for q in self.m_q_list:
                    q = float(q)

                    # q=0, q=1 are handled above (kind="N"/"S"); skip
                    # them in the logMq pass even if the user listed
                    # them, to avoid duplicate rows.
                    if np.isclose(q, 1.0) or np.isclose(q, 0.0):
                        continue

                    # log M(q, eps) = log sum_i mu_i^q
                    # log_mu_all is finite (mu_pos_all > 0), so q*log_mu_all
                    # is finite for any finite q. logsumexp is numerically
                    # stable here.
                    a = q * log_mu_all
                    logMq = float(logsumexp(a)) if a.size else np.nan
                    if np.isfinite(logMq):
                        records.append({"scale": size, "eps": eps, "q": q,
                                        "value": logMq, "kind": "logMq"})
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

        # --- q != 0,1: use y = logMq directly ---
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

            tau = float(-slope)
            Dq = float(tau / (qv - 1.0))
            out.append({"q": qv, "tau": tau, "Dq": Dq, "D1": np.nan,
                        "intercept": float(intercept), "r_value": float(r_value),
                        "p_value": float(p_value), "std_err": float(std_err),
                        "n_points": int(n_used)})

        # --- q == 0: use y = log N ---
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

                D1 = float(slope)
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
                use_middle_scales=False, fit_scale_frac=(0.2, 0.8),
                if_auto_line_fit=False, auto_fit_min_len_ratio=0.5,
                spline_s=0, cap_d0_at_2=False):
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
    # Batch processing of multiple images
    # ------------------------------------------------------------
    @staticmethod
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
                      with_progress=True):
        """
        Batch CPU multifractal spectrum.

        Returns
        -------
        list of (df_mass, df_fit, df_spec) tuples, one per input image.

        Notes
        -----
        For CPU we run images independently on the same thread; the
        per-image cost is dominated by per-scale view_as_blocks ops,
        which are already vectorised inside `get_mass_table`. This
        wrapper exists to give the CPU class an API parity with
        CFA2DMFSGPU.get_batch_mfs.
        """
        if len(img_list) == 0:
            return []
        results = []
        iterator = (tqdm(enumerate(img_list), total=len(img_list),
                          desc="Batch MFS (CPU)")
                     if with_progress else enumerate(img_list))
        for i, img in iterator:
            obj = CFAImageMFS(image=img,
                            corp_type=corp_type,
                            q_list=q_list,
                            with_progress=False,
                            bg_threshold=bg_threshold,
                            bg_reverse=bg_reverse,
                            bg_otsu=bg_otsu,
                            mu_floor=mu_floor)
            df_mass, df_fit, df_spec = obj.get_mfs(
                max_size=max_size, max_scales=max_scales,
                min_points=min_points, min_box=min_box,
                use_middle_scales=use_middle_scales,
                fit_scale_frac=fit_scale_frac,
                if_auto_line_fit=if_auto_line_fit,
                auto_fit_min_len_ratio=auto_fit_min_len_ratio,
                spline_s=spline_s, cap_d0_at_2=cap_d0_at_2)
            results.append((df_mass, df_fit, df_spec))
        return results

    # ------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------
    def plot(self, df_mass, df_fit, df_spec):
        fig, axs = plt.subplots(2, 3, figsize=(14,8))

        # Heatmap: logMq for q!=0,1
        df_logmq = df_mass[df_mass["kind"] == "logMq"].copy()
        if not df_logmq.empty:
            pivot = df_logmq.pivot_table(index="scale", columns="q", values="value", aggfunc="mean")
            vals = pivot.values
            if np.any(np.isfinite(vals)):
                vmin = np.nanpercentile(vals, 10)
                vmax = np.nanpercentile(vals, 90)
                sns.heatmap(pivot, ax=axs[0, 0], cmap="coolwarm", vmin=vmin, vmax=vmax, cbar=True)
                axs[0, 0].set_xlabel("$q$")
                axs[0, 0].set_ylabel("box size (pixels)")
                axs[0, 0].set_title("Heatmap: log M(q, ε) vs box size and q")
                axs[0, 0].xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
                axs[0, 0].tick_params(axis='x', labelrotation=30,labelsize=8)
                axs[0, 0].tick_params(axis='y', labelrotation=30,labelsize=8)
            else:
                axs[0, 0].set_title("Heatmap: (all NaN)")
        else:
            axs[0, 0].set_title("Heatmap: (no logMq data)")

        if df_spec is not None and not df_spec.empty and np.any(np.isfinite(df_spec["alpha"].values)):
            sns.lineplot(data=df_spec, x="alpha", y="f_alpha", ax=axs[0, 1])
            axs[0, 1].set_xlabel(r"$\alpha$")
            axs[0, 1].set_ylabel(r"$f(\alpha)$")
            axs[0, 1].set_title(r"Multifractal spectrum: $f(\alpha)$ vs $\alpha$")
            axs[0, 1].grid(True)
            axs[0, 1].tick_params(axis='x', labelrotation=30,labelsize=8)
            axs[0, 1].tick_params(axis='y', labelrotation=30,labelsize=8)
        else:
            axs[0, 1].set_title("f(alpha): (no data)")

        if df_fit is not None and not df_fit.empty:
            sns.lineplot(data=df_fit, x="q", y="tau", ax=axs[0, 2])
            axs[0, 2].set_xlabel("$q$")
            axs[0, 2].set_ylabel(r"$\tau(q)$")
            axs[0, 2].set_title(r"$\tau(q)$ vs $q$")
            axs[0, 2].grid(True)
            #axs[0, 2].tick_params(axis='x', labelrotation=30,labelsize=8)
            axs[0, 2].tick_params(axis='y', labelrotation=30,labelsize=8)
        else:
            axs[0, 2].set_title("tau(q): (no data)")

        if df_fit is not None and not df_fit.empty:
            sns.lineplot(data=df_fit, x="q", y="Dq", ax=axs[1, 0])
            axs[1, 0].set_xlabel("$q$")
            axs[1, 0].set_ylabel(r"$D(q)$")
            axs[1, 0].set_title(r"$D(q)$ vs $q$ (with $D_1$ at q=1)")
            axs[1, 0].grid(True)
            #axs[1, 0].tick_params(axis='x', labelrotation=30,labelsize=8)
            axs[1, 0].tick_params(axis='y', labelrotation=30,labelsize=8)
        else:
            axs[1, 0].set_title("D(q): (no data)")

        if df_spec is not None and not df_spec.empty and np.any(np.isfinite(df_spec["alpha"].values)):
            sns.lineplot(data=df_spec, x="q", y="alpha", ax=axs[1, 1])
            axs[1, 1].set_xlabel(r"$q$")
            axs[1, 1].set_ylabel(r"$\alpha$")
            axs[1, 1].set_title(r"Multifractal spectrum: $\alpha$ vs $q$")
            axs[1, 1].grid(True)
            #axs[1, 1].tick_params(axis='x', labelrotation=30,labelsize=8)
            axs[1, 1].tick_params(axis='y', labelrotation=30,labelsize=8)
        else:
            axs[1, 1].set_title("f(alpha): (no data)")

        if df_spec is not None and not df_spec.empty and np.any(np.isfinite(df_spec["f_alpha"].values)):
            sns.lineplot(data=df_spec, x="q", y="f_alpha", ax=axs[1, 2])
            axs[1, 2].set_xlabel(r"$q$")
            axs[1, 2].set_ylabel(r"$f(\alpha)$")
            axs[1, 2].set_title(r"Multifractal spectrum: $f(\alpha)$ vs $q$")
            axs[1, 2].grid(True)
            #axs[1, 2].tick_params(axis='x', labelrotation=30,labelsize=8)
            axs[1, 2].tick_params(axis='y', labelrotation=30,labelsize=8)
        else:
            axs[1, 2].set_title("f(alpha): (no data)")
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_alpha_map(alpha_map):
        """
        Visualize alpha(x,y)
        """
        plt.figure(figsize=(6, 6))
        plt.imshow(alpha_map, cmap="jet")
        plt.colorbar(label=r"$\alpha(x,y)$")
        plt.title("Local Multifractal α-map")
        plt.axis("off")
        plt.show()
    # ============================================================
    # Local (coarse-grained) singularity map alpha(x, y)
    # ============================================================
    #
    # Definition
    # ----------
    # For a probability measure mu derived from the image, the local
    # singularity (Holder) exponent at pixel x is
    #
    #     mu(B_eps(x)) ~ eps^alpha(x)   as eps -> 0
    #
    # so log mu(eps, x) is linear in log(eps) with slope alpha(x). We
    # estimate alpha(x) per pixel by:
    #
    #   1. For each scale s in `scales`:
    #        - tile the fixed L x L ROI into (L//s) x (L//s) boxes,
    #        - compute box probabilities mu_i (sums to 1 over boxes),
    #        - "broadcast" each box value to the s x s pixels it covers,
    #          producing a per-pixel measure mu_s(x,y) of shape (L, L).
    #   2. Stack mu_s across scales -> mu_stack of shape (n_scales, L, L).
    #   3. At every pixel (x, y), regress log mu_s(x, y) on log eps_s and
    #      take the slope. This is done analytically (cov / var) so the
    #      whole map is one vectorized op, no per-pixel linregress call.
    #
    # Why this differs from the original snippet
    # ------------------------------------------
    # The original code:
    #   - Used `i * scales[k] // size`, which mixes up the direction of
    #     the scale-to-grid index map (correct map is
    #     `(i*size) // scales[k]`).
    #   - Looped the SAME `len(scales)`-point trajectory inside an outer
    #     `for size in scales` loop, then *overwrote* alpha_map at each
    #     pixel several times, so the final value depended on iteration
    #     order, not on a real average.
    #   - Took log(mu + 1e-12), pulling alpha down at empty boxes.
    #   - Used scipy.stats.linregress per pixel, which is slow.
    #
    # The new implementation:
    #   - Computes alpha analytically per pixel via cov/var (vectorized).
    #   - Treats empty-box pixels (mu == 0 at any scale) as missing in
    #     log space and drops them from the regression on a per-pixel
    #     basis, so empty regions become NaN rather than -27.6 outliers.
    # ============================================================
    def _scale_mu_grid(self, img_fixed, size):
        """
        Compute the (nY, nX) box probabilities for a single scale on a
        fixed L x L ROI, ignoring corp_type (we want exact tiling here).
        """
        L = img_fixed.shape[0]
        new_L = (L // size) * size
        if new_L == 0:
            return None
        sub = img_fixed[:new_L, :new_L]
        n = new_L // size
        # block-sum: (n, size, n, size) -> (n, n)
        bm = sub.reshape(n, size, n, size).sum(axis=(1, 3))
        bm = np.clip(bm, 0.0, None)
        total = float(bm.sum())
        if (not np.isfinite(total)) or total <= 0:
            return None
        mu = bm / total
        return mu  # (n, n), aligned to the top-left (new_L x new_L) region

    def compute_alpha_map(self, scales=None, roi_mode="center",
                          empty_policy="nan"):
        """
        Compute the per-pixel local singularity exponent map alpha(x, y).

        Parameters
        ----------
        scales : iterable of int or None
            Box sizes (in pixels). Default: powers of two up to L/4.
        roi_mode : {"center", "topleft"}
            How to crop the image to a square L x L ROI.
        empty_policy : {"nan", "fill"}
            How to handle pixels where some scale gives mu == 0:
              * "nan"  -> drop those (scale, pixel) entries; if a pixel
                          has fewer than 2 valid scales, alpha = NaN.
              * "fill" -> replace 0 with the smallest positive mu observed
                          at that scale before taking log.

        Returns
        -------
        alpha_map : (L, L) float64
            NaN where alpha cannot be estimated.
        info : dict
            {"L": L, "scales": np.ndarray, "log_eps": np.ndarray}
        """
        # Reuse the batch path with B=1 (streaming OLS, low memory).
        maps, info = CFAImageMFS.compute_alpha_map_batch(
            [self.m_image], scales=scales, roi_mode=roi_mode,
            bg_threshold=0.0,             # already preprocessed in __init__
            bg_reverse=False, bg_otsu=False, empty_policy=empty_policy,
            with_progress=False)
        return maps[0], info

    @staticmethod
    def compute_alpha_map_batch(images, scales=None, roi_mode="center",
                                bg_threshold=0.01, bg_reverse=False,
                                bg_otsu=False, empty_policy="nan",
                                with_progress=True, mu_floor=1e-12):
        """
        Batch version of compute_alpha_map for a list of grayscale images.

        Implementation: grid-level streaming OLS. Two key optimisations:

          1) Nested-grid trick. When every scale s_k is a multiple of the
             smallest scale s_min, every s_min x s_min pixel patch has a
             constant alpha (because the box trajectories at the two
             pixels in that patch are identical at every scale). We
             therefore run OLS on the (L/s_min, L/s_min) coarse grid and
             upsample once at the end. This shrinks the OLS buffers by
             a factor of s_min^2 (e.g. 4x for s_min=2, 256x for s_min=16).

          2) Streaming OLS. We never materialise the (B, n_scales, L, L)
             tensor; we accumulate the five ordinary-least-squares sums
             [n_eff, sum_x, sum_y, sum_xx, sum_xy] over scales.

          When the scales are NOT all integer multiples of s_min, we fall
          back to pixel-level streaming OLS (still no full mu_stack).

        Returns
        -------
        alpha_maps : list of (L, L) float64
        info : dict with the common L, scales, log_eps.
        """
        if len(images) == 0:
            return [], {"L": 0, "scales": np.array([], dtype=int),
                        "log_eps": np.array([], dtype=np.float64)}

        def _prep(img):
            tmp = CFAImageMFS(image=img, corp_type=-1,
                           q_list=np.array([0.0]),
                           with_progress=False,
                           bg_threshold=bg_threshold,
                           bg_reverse=bg_reverse,
                           bg_otsu=bg_otsu,
                           mu_floor=mu_floor)
            return tmp.m_image

        prepped = [_prep(im) for im in images]
        L_common = int(min(min(im.shape) for im in prepped))

        rois = np.stack(
            [crop_square_roi(im, L=L_common, mode=roi_mode)[0]
             for im in prepped], axis=0
        ).astype(np.float64, copy=False)
        B = rois.shape[0]

        if scales is None:
            max_pow = max(2, int(np.floor(np.log2(max(2, L_common // 4)))))
            scales = np.array([2 ** k for k in range(1, max_pow + 1)
                               if 2 ** k <= L_common], dtype=int)
        scales = np.array(sorted({int(s) for s in scales
                                  if 1 <= s <= L_common}), dtype=int)
        if scales.size < 2:
            raise ValueError("Need at least 2 valid scales for alpha map.")
        log_eps = np.log(scales.astype(np.float64) / float(L_common))

        # Nested-grid optimisation: if every s is a multiple of s_min and
        # L_common is a multiple of s_min, the alpha map is constant on
        # every s_min x s_min patch, so we can do OLS on the coarse grid.
        s_min = int(scales.min())
        nested_ok = (L_common % s_min == 0) and \
                    bool(np.all((scales % s_min) == 0))

        if nested_ok:
            grid_L = L_common // s_min                         # coarse side
            n_eff  = np.zeros((B, grid_L, grid_L), dtype=np.float64)
            sum_x  = np.zeros_like(n_eff)
            sum_y  = np.zeros_like(n_eff)
            sum_xx = np.zeros_like(n_eff)
            sum_xy = np.zeros_like(n_eff)
        else:
            # Fall back to pixel grid (rare path).
            grid_L = L_common
            n_eff  = np.zeros((B, grid_L, grid_L), dtype=np.float64)
            sum_x  = np.zeros_like(n_eff)
            sum_y  = np.zeros_like(n_eff)
            sum_xx = np.zeros_like(n_eff)
            sum_xy = np.zeros_like(n_eff)

        iter_scales = (tqdm(enumerate(scales), total=scales.size,
                            desc="Batch alpha-map (CPU streaming)")
                       if with_progress else enumerate(scales))

        for k, s in iter_scales:
            s = int(s)
            x_k = float(log_eps[k])
            new_L = (L_common // s) * s
            if new_L == 0:
                continue
            n = new_L // s
            sub = rois[:, :new_L, :new_L]
            bm = sub.reshape(B, n, s, n, s).sum(axis=(2, 4))    # (B,n,n)
            np.clip(bm, 0.0, None, out=bm)
            total = bm.reshape(B, -1).sum(axis=1)               # (B,)
            ok = np.isfinite(total) & (total > 0)
            if not np.any(ok):
                continue
            safe = np.where(ok, total, 1.0)[:, None, None]
            mu = bm / safe                                      # (B,n,n)
            mu[~ok] = 0.0

            if empty_policy == "fill":
                mu_pos = np.where(mu > 0, mu, np.inf)
                mins = mu_pos.reshape(B, -1).min(axis=1)
                mins = np.where(np.isfinite(mins), mins, 1e-300)
                mu_filled = np.where(mu > 0, mu, mins[:, None, None])
                with np.errstate(divide="ignore"):
                    log_mu_grid = np.log(mu_filled)
                valid_grid = np.broadcast_to(
                    ok[:, None, None], (B, n, n)).astype(np.float64)
            else:
                pos = (mu > 0)
                with np.errstate(divide="ignore"):
                    log_mu_grid = np.where(pos, np.log(np.where(pos, mu, 1.0)),
                                           0.0)
                valid_grid = (pos & ok[:, None, None]).astype(np.float64)

            # Upsample (B, n, n) to (B, grid_L, grid_L) (or sub-region).
            if nested_ok:
                # On the coarse grid, n divides grid_L exactly:
                # grid_L = L/s_min, n = L/s -> rep = grid_L / n = s/s_min
                rep = s // s_min
                log_mu_up = np.repeat(np.repeat(log_mu_grid, rep, axis=1),
                                      rep, axis=2)
                valid_up = np.repeat(np.repeat(valid_grid, rep, axis=1),
                                     rep, axis=2)
                # On the coarse grid the box covers all rows/cols, no
                # sub-region needed because new_L / s_min = n * rep = grid_L
                # only when (L_common - new_L) < s_min. If new_L < L_common,
                # the un-tiled trailing strip on the pixel grid corresponds
                # to floor((L_common - new_L) / s_min) coarse cells. We
                # restrict the update to grid_L_new = (new_L // s_min):
                gL = new_L // s_min
                n_eff [:, :gL, :gL] += valid_up[:, :gL, :gL]
                sum_x [:, :gL, :gL] += valid_up[:, :gL, :gL] * x_k
                yt = np.where(valid_up[:, :gL, :gL] > 0,
                              log_mu_up[:, :gL, :gL], 0.0)
                sum_y [:, :gL, :gL] += yt
                sum_xx[:, :gL, :gL] += valid_up[:, :gL, :gL] * (x_k * x_k)
                sum_xy[:, :gL, :gL] += yt * x_k
            else:
                # Pixel-grid path (slower; only used when scales are not
                # all multiples of s_min).
                log_mu_pix = np.repeat(np.repeat(log_mu_grid, s, axis=1),
                                       s, axis=2)
                valid_pix = np.repeat(np.repeat(valid_grid, s, axis=1),
                                       s, axis=2)
                n_eff [:, :new_L, :new_L] += valid_pix
                sum_x [:, :new_L, :new_L] += valid_pix * x_k
                yt = np.where(valid_pix > 0, log_mu_pix, 0.0)
                sum_y [:, :new_L, :new_L] += yt
                sum_xx[:, :new_L, :new_L] += valid_pix * (x_k * x_k)
                sum_xy[:, :new_L, :new_L] += yt * x_k

        denom = n_eff * sum_xx - sum_x * sum_x
        numer = n_eff * sum_xy - sum_x * sum_y
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.where((n_eff >= 2) & (np.abs(denom) > 0),
                             numer / denom, np.nan)

        if nested_ok and s_min > 1:
            # upsample (B, grid_L, grid_L) -> (B, L, L) by nearest repeat
            slope_full = np.repeat(np.repeat(slope, s_min, axis=1),
                                   s_min, axis=2)
            # The trailing strip (if L_common is not exactly s_min * grid_L)
            # can never happen because we required (L_common % s_min == 0).
            return [slope_full[i].astype(np.float64) for i in range(B)], \
                   {"L": L_common, "scales": scales, "log_eps": log_eps}
        else:
            return [slope[i].astype(np.float64) for i in range(B)], \
                   {"L": L_common, "scales": scales, "log_eps": log_eps}


# ============================================================
# Vectorized per-pixel slope: log mu(eps) vs log eps
# ============================================================
def _alpha_map_from_mu_stack(mu_stack, log_eps, empty_policy="nan"):
    """
    Compute slope of log mu_stack[k, :, :] vs log_eps[k], independently
    at every pixel, in a single vectorized pass.

    NaN in mu_stack (e.g., outside the per-scale aligned region) is
    treated as missing for that pixel's regression. Same for mu == 0:

      * empty_policy="nan":  drop those (scale, pixel) entries; if a
                              pixel has < 2 valid scales, alpha = NaN.
      * empty_policy="fill": replace mu == 0 with min positive mu at
                              that scale before taking log.

    Returns
    -------
    alpha_map : (L, L) float64, NaN where undetermined.
    """
    K = mu_stack.shape[0]
    L = mu_stack.shape[1]

    if empty_policy == "fill":
        ms = mu_stack.copy()
        for k in range(K):
            slab = ms[k]
            pos = slab[(slab > 0) & np.isfinite(slab)]
            floor = float(pos.min()) if pos.size else 1e-300
            slab = np.where((slab > 0) & np.isfinite(slab), slab, floor)
            ms[k] = slab
        log_mu = np.log(ms)
        valid = np.ones_like(log_mu, dtype=bool)  # everything valid
    else:
        # "nan": only positive, finite entries are valid
        ms = mu_stack
        with np.errstate(divide="ignore", invalid="ignore"):
            log_mu = np.where(np.isfinite(ms) & (ms > 0),
                              np.log(np.where(ms > 0, ms, 1.0)),
                              np.nan)
        valid = np.isfinite(log_mu)

    # x[k]: log_eps[k]; per-pixel weighted least squares with weights
    # being the 0/1 valid mask.
    x = log_eps.astype(np.float64).reshape(K, 1, 1)                # (K,1,1)
    y = log_mu                                                     # (K,L,L)
    w = valid.astype(np.float64)                                   # (K,L,L)

    n_eff   = w.sum(axis=0)                                        # (L,L)
    sum_x   = (w * x).sum(axis=0)
    sum_y   = np.where(valid, y, 0.0).sum(axis=0)
    sum_xx  = (w * x * x).sum(axis=0)
    sum_xy  = (np.where(valid, y, 0.0) * x).sum(axis=0)

    denom = n_eff * sum_xx - sum_x * sum_x
    numer = n_eff * sum_xy - sum_x * sum_y

    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where((n_eff >= 2) & (np.abs(denom) > 0),
                         numer / denom, np.nan)
    return slope.astype(np.float64)


# ============================================================
# Demo
# ============================================================
def main():
    image_path = "../images/fractal.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)
    with_progress = True
    q_list = np.linspace(-5, 5, 51)
    mfs = CFAImageMFS(image=image,
                      corp_type=-1,
                      q_list=q_list,
                      with_progress=with_progress,
                      bg_reverse=False,
                      bg_threshold=0.01,
                      bg_otsu=False)

    print(mfs.m_image)
    print("Nonzero pixel ratio (adjust bg_threshold, typically < 0.5):", np.mean(mfs.m_image > 0))

    df_mass, df_fit, df_spec = mfs.get_mfs(max_scales=80,
                                           min_points=6,
                                           use_middle_scales=False,
                                           if_auto_line_fit=False,
                                           fit_scale_frac=(0.3, 0.7),
                                           auto_fit_min_len_ratio=0.6,
                                           cap_d0_at_2=False)
    print(df_fit.head())

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

    alpha_map, info = mfs.compute_alpha_map(scales=[2, 4, 8, 16, 32])
    print(alpha_map)
    print(info)
    mfs.plot_alpha_map(alpha_map)

if __name__ == "__main__":
    main()
