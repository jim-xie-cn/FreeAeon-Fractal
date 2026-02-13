import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import time
import torch
from scipy.stats import linregress
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker


# ============================================================
# Fixed ROI utilities (Scheme A: fixed square ROI, no LCM)
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
# GPU-accelerated MFS
# ============================================================
class CFA2DMFSGPU:
    """
    GPU version of CFA2DMFS (PyTorch).
    - Core per-scale μ and logsumexp computed on GPU.
    - Regression/spline kept on CPU.
    """

    def __init__(self, image, corp_type=0, q_list=np.linspace(-5, 5, 51),
                 with_progress=True, bg_threshold=0.01, bg_reverse=False,
                 bg_otsu=False, mu_floor=1e-12, device='cuda', dtype=torch.float64):

        if image is None:
            raise ValueError("image is None")
        if image.ndim != 2:
            raise ValueError("image must be a 2D grayscale array (H,W).")

        img_raw = image.astype(np.float64)

        # optional Otsu on raw image (CPU)
        if bg_otsu:
            vmin = np.nanmin(img_raw)
            vmax = np.nanmax(img_raw)
            if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                img8 = ((img_raw - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
                _, img_bin = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                img_raw[img_bin == 0] = 0.0

        # Normalize to [0,1] (CPU)
        img = img_raw.copy()
        img -= np.nanmin(img)
        mx = np.nanmax(img)
        img /= (mx + 1e-12)
        img = np.where(np.isfinite(img), img, 0.0)
        img = np.clip(img, 0.0, 1.0)

        # background thresholding (CPU)
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
        self._mu_floor = mu_floor  # kept for compatibility, not used

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype

    # ------------------------------------------------------------
    # Core: μ at a given box size (GPU)
    # ------------------------------------------------------------
    @torch.no_grad()
    def _mu_at_scale_torch(self, img_t: torch.Tensor, box_size: int):
        """
        img_t: (H,W) on device
        returns mu_pos (1D tensor, mu>0), and also log(mu_pos)
        """
        H, W = img_t.shape
        s = int(box_size)
        if s <= 0 or H % s != 0 or W % s != 0:
            # Scheme A expects exact tiling
            return None, None

        # (H,W) -> (nY, s, nX, s) -> sum over block dims -> (nY,nX) -> flatten
        nY = H // s
        nX = W // s
        blocks = img_t.reshape(nY, s, nX, s)
        block_mass = blocks.sum(dim=(1, 3)).reshape(-1)  # (nY*nX,)

        total = block_mass.sum()
        if not torch.isfinite(total) or total <= 0:
            return None, None

        mu = block_mass / total
        # keep only mu>0 to avoid log(0) issues (same as your code)
        mu_pos = mu[mu > 0]
        if mu_pos.numel() == 0:
            return None, None

        log_mu = torch.log(mu_pos)
        return mu_pos, log_mu

    # ------------------------------------------------------------
    # Per-scale table (GPU)
    # ------------------------------------------------------------
    @torch.no_grad()
    def get_mass_table(self, max_size=None, max_scales=80, min_box=2, roi_mode="center"):
        img0 = self.m_image
        h0, w0 = img0.shape
        if max_size is None:
            max_size = min(h0, w0)
        if max_size < min_box:
            raise ValueError("max_size too small.")

        # Scheme A fixed ROI (CPU crop)
        img_fixed_np, fixed_L_int = crop_square_roi(
            img0, L=min(min(h0, w0), max_size), mode=roi_mode
        )
        fixed_L = float(fixed_L_int)

        # Candidate scales (CPU)
        scales = np.logspace(np.log2(min_box), np.log2(fixed_L_int), num=max_scales, base=2.0)
        scales = np.unique(np.maximum(min_box, np.round(scales).astype(int)))
        scales = scales[(scales >= min_box) & (scales <= fixed_L_int)]
        scales = scales[(fixed_L_int % scales) == 0]
        scales = np.array(sorted(scales.astype(int)))
        if scales.size == 0:
            return pd.DataFrame()

        # Move fixed ROI to GPU once
        img_t = torch.from_numpy(img_fixed_np).to(device=self.device, dtype=self.dtype)

        # q on GPU
        q_t = torch.tensor(self.m_q_list, device=self.device, dtype=self.dtype)

        records = []
        iterator = tqdm(scales, desc=f"Computing per-scale μ on {self.device.type}") if self.m_with_progress else scales

        for size in iterator:
            size = int(size)
            mu_pos, log_mu = self._mu_at_scale_torch(img_t, size)
            if mu_pos is None:
                continue

            eps = float(size) / fixed_L

            # q==0: N = count(mu>0)
            # q==1: S = -sum(mu log mu)
            # q!=0,1: logMq = logsumexp(q*log(mu))
            N = float(mu_pos.numel())
            S = float(-(mu_pos * log_mu).sum().item())

            # 先把 q==0/1 写入（CPU records）
            records.append({"scale": size, "eps": eps, "q": 0.0, "value": N, "kind": "N"})
            records.append({"scale": size, "eps": eps, "q": 1.0, "value": S, "kind": "S"})

            # 计算所有 q 的 logMq（GPU），然后过滤掉 q=0/1
            # a shape: (n_mu, n_q) if we broadcast properly
            # we want for each q: logsumexp(q*log_mu)
            a = log_mu[:, None] * q_t[None, :]  # (n_mu, n_q)
            logMq_all = torch.logsumexp(a, dim=0)  # (n_q,)

            # 搬回 CPU
            logMq_cpu = logMq_all.detach().to("cpu").numpy().astype(np.float64)

            for qv, logMq in zip(self.m_q_list.astype(np.float64), logMq_cpu):
                if np.isclose(qv, 0.0) or np.isclose(qv, 1.0):
                    continue
                if np.isfinite(logMq):
                    records.append({"scale": size, "eps": eps, "q": float(qv), "value": float(logMq), "kind": "logMq"})

        df = pd.DataFrame(records)
        if df.empty:
            return df
        return df.sort_values(["kind", "q", "scale"]).reset_index(drop=True)

    # ------------------------------------------------------------
    # 下面：拟合/谱计算/画图基本保持你原来的 CPU 实现
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

    def fit_tau_and_D1(
        self, df_mass, min_points=6, require_common_scales=True,
        use_middle_scales=True, fit_scale_frac=(0.2, 0.8),
        if_auto_line_fit=False, auto_fit_min_len_ratio=0.5,
        cap_d0_at_2=False
    ):
        # 原逻辑基本不变（CPU）
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

        # q != 0,1
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

        # q==0
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
                            idx = np.argmax(xs)
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

                tau0 = float(-slope)
                D0 = float(slope)
                out.append({"q": 0.0, "tau": tau0, "Dq": D0, "D1": np.nan,
                            "intercept": float(intercept), "r_value": float(r_value),
                            "p_value": float(p_value), "std_err": float(std_err),
                            "n_points": int(n_used)})
            else:
                out.append({"q": 0.0, "tau": np.nan, "Dq": np.nan, "D1": np.nan,
                            "intercept": np.nan, "r_value": np.nan, "p_value": np.nan,
                            "std_err": np.nan, "n_points": int(x0.size)})

        # q==1
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

    def plot(self, df_mass, df_fit, df_spec):
        fig, axs = plt.subplots(2, 2, figsize=(8, 6))

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

def main():
    image_path = "../images/fractal.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)

    q_list = np.linspace(0, 5, 51)
    mfs = CFA2DMFSGPU(
            image=image,
            corp_type=0,
            q_list=q_list,
            with_progress=True,
            bg_reverse=False,
            bg_threshold=0.01,
            bg_otsu=False)
    df_mass, df_fit, df_spec = mfs.get_mfs(
            max_scales=80,
            min_points=6,
            use_middle_scales=False,
            if_auto_line_fit=False,
            fit_scale_frac=(0.3, 0.7),
            auto_fit_min_len_ratio=0.6,
            cap_d0_at_2=False)

    print(df_fit.head())
    mfs.plot(df_mass, df_fit, df_spec)

if __name__ == "__main__":
    main()

