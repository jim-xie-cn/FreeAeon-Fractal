"""
Improved Multifractal (box-counting) analysis for 2D grayscale images.

Fixes:
- q=1 entropy uses original μ (no floor, no log_eps).
- Negative q stability via logsumexp and μ floor only for q!=1.
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

        if corp_type == -1:
            img = CFAImage.crop_data(image, block_size)
        elif corp_type == 0:
            img = image
        else:
            img = CFAImage.crop_data(image, block_size)

        bh, bw = block_size
        if img.shape[0] < bh or img.shape[1] < bw:
            return np.empty((0, bh, bw), dtype=img.dtype), np.empty((0, 0, bh, bw), dtype=img.dtype)

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
        -1 crop to multiples of box size (recommended).
    q_list : array-like
        q values. Can include 1; we compute D1 specially.
    with_progress : bool
        Show progress bars.
    """

    def __init__(self, image, corp_type=-1, q_list=np.linspace(-5, 5, 51), with_progress=True):
        if image is None:
            raise ValueError("image is None")
        if image.ndim != 2:
            raise ValueError("image must be a 2D grayscale array (H,W).")

        img = image.astype(np.float64)

        # Normalize to [0,1] and ensure nonnegative measure
        img -= np.nanmin(img)
        mx = np.nanmax(img)
        img /= (mx + 1e-12)
        img = np.where(np.isfinite(img), img, 0.0)
        img = np.clip(img, 0.0, 1.0)

        self.m_image = img
        self.m_corp_type = corp_type
        self.m_with_progress = with_progress
        self.m_q_list = np.array(q_list, dtype=np.float64)

        # Small epsilon only used inside logs for q!=1 (not for changing μ algebra)
        self._log_eps = 1e-300
        self._mu_floor = 1e-300  # for log stability with negative q

    # ------------------------------------------------------------
    # Core: compute per-scale measures μ_i(ε)
    # ------------------------------------------------------------
    def _mu_at_scale(self, box_size):
        """
        Compute box probabilities μ_i at a given box size.

        Returns:
            mu : 1D float64 array, μ_i >= 0, sum(mu)=1 (on cropped region)
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
    # Compute M(q,ε) or S(ε) per scale
    # ------------------------------------------------------------
    def get_mass_table(self, max_size=None, max_scales=80, min_box=2):
        """
        Compute per-(scale,q) table needed for τ(q) and D1.

        Output columns:
            scale, eps, q, value, kind
        where:
            kind = "Mq" for q!=1 (value = Σ μ^q or N_nonzero for q==0)
            kind = "S"  for q==1 (value = -Σ μ log μ)
        """
        img = self.m_image
        h, w = img.shape
        if max_size is None:
            max_size = min(h, w)

        if max_size < min_box:
            raise ValueError("max_size too small.")

        scales = np.logspace(np.log2(min_box), np.log2(max_size), num=max_scales, base=2.0)
        scales = np.unique(np.maximum(min_box, np.round(scales).astype(int)))

        records = []
        iterator = tqdm(scales, desc="Computing per-scale μ") if self.m_with_progress else scales

        for size in iterator:
            if size < min_box:
                continue

            mu = self._mu_at_scale(size)
            if mu.size == 0:
                continue

            mu_pos = mu[mu > 0]
            if mu_pos.size == 0:
                continue

            eps = size / float(min(h, w))  # normalized scale

            for q in self.m_q_list:
                if np.isclose(q, 1.0):
                    # IMPORTANT: no floor or log_eps for q=1
                    log_mu = np.log(mu_pos)
                    S = float(-np.sum(mu_pos * log_mu))
                    if np.isfinite(S):
                        records.append({"scale": int(size), "eps": eps, "q": 1.0, "value": S, "kind": "S"})
                    continue

                if np.isclose(q, 0.0):
                    records.append({"scale": int(size), "eps": eps, "q": 0.0, "value": float(mu_pos.size), "kind": "Mq"})
                    continue

                # q != 1: stabilize negative q
                mu_floor = np.maximum(mu_pos, self._mu_floor)
                log_mu_floor = np.log(mu_floor + self._log_eps)

                logMq = float(logsumexp(q * log_mu_floor))
                if not np.isfinite(logMq):
                    continue
                if logMq > 700:
                    continue
                Mq = float(np.exp(logMq))
                if np.isfinite(Mq) and Mq > 0:
                    records.append({"scale": int(size), "eps": eps, "q": float(q), "value": Mq, "kind": "Mq"})

        df = pd.DataFrame(records)
        if df.empty:
            return df
        return df.sort_values(["kind", "q", "scale"]).reset_index(drop=True)

    # ------------------------------------------------------------
    # Helper: choose a common scale set for regression
    # ------------------------------------------------------------
    @staticmethod
    def _common_scales_for_fit(df_mass, require_q1=True):
        """
        Return a sorted numpy array of scales that are present for:
          - every q in Mq-kind (q!=1), and
          - (optionally) q=1 in S-kind.
        """
        df_mq = df_mass[df_mass["kind"] == "Mq"].copy()
        if df_mq.empty:
            return np.array([], dtype=int)

        common = None
        for q, g in df_mq.groupby("q"):
            scales_q = set(g["scale"].astype(int).tolist())
            common = scales_q if common is None else (common & scales_q)

        if common is None:
            return np.array([], dtype=int)

        if require_q1:
            df_s = df_mass[(df_mass["kind"] == "S") & (np.isclose(df_mass["q"], 1.0))].copy()
            if df_s.empty:
                return np.array([], dtype=int)
            common = common & set(df_s["scale"].astype(int).tolist())

        return np.array(sorted(common), dtype=int)

    # ------------------------------------------------------------
    # Fit τ(q) and D1 (consistent scales + x=log(1/ε))
    # ------------------------------------------------------------
    def fit_tau_and_D1(self, df_mass, min_points=6, require_common_scales=True):
        """
        Fit using a COMMON scale set for all q:
            - For q != 1:
                log M(q,ε) ~ -τ(q) * log(1/ε) + c
                so if x = log(1/ε), slope = d logM / d x = -τ(q)  => τ(q) = -slope
            - For q == 1:
                S(ε) ~ D1 * log(1/ε) + c
                so D1 is directly the slope (no minus sign)

            - For q == 0 (where value=N(ε)):
                log N(ε) ~ D0 * log(1/ε) + c
                so D0 is directly the slope; do NOT use tau/(q-1).
        """
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

        def _filter_common(d):
            if common_scales is None:
                return d
            return d[d["scale"].isin(common_scales)].copy()

        out = []

        # ---- q != 1: tau(q) from log Mq vs log(1/eps)
        df_mq = _filter_common(df_mass[df_mass["kind"] == "Mq"])
        for q, df_q in df_mq.groupby("q"):
            qv = float(q)
            if np.isclose(qv, 1.0):
                continue

            eps = df_q["eps"].astype(np.float64).values
            x = np.log(1.0 / eps)
            y = np.log(df_q["value"].astype(np.float64).values)

            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]

            if x.size < min_points:
                out.append({"q": qv, "tau": np.nan, "Dq": np.nan, "D1": np.nan,
                            "intercept": np.nan, "r_value": np.nan, "p_value": np.nan,
                            "std_err": np.nan, "n_points": int(x.size)})
                continue

            slope, intercept, r_value, p_value, std_err = linregress(x, y)

            tau = float(-slope)
            if np.isclose(qv, 0.0):
                Dq = float(slope)  # D0 = slope
            else:
                Dq = float(tau / (qv - 1.0))

            out.append({"q": qv, "tau": tau, "Dq": Dq, "D1": np.nan,
                        "intercept": float(intercept), "r_value": float(r_value),
                        "p_value": float(p_value), "std_err": float(std_err),
                        "n_points": int(x.size)})

        # ---- q == 1: D1 from S(ε) vs log(1/eps)
        df_s = _filter_common(df_mass[(df_mass["kind"] == "S") & (np.isclose(df_mass["q"], 1.0))])
        if not df_s.empty:
            eps = df_s["eps"].astype(np.float64).values
            x = np.log(1.0 / eps)
            y = df_s["value"].astype(np.float64).values  # entropy S(ε)

            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]

            if x.size >= min_points:
                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                D1 = float(slope)
                out.append({"q": 1.0, "tau": 0.0, "Dq": D1, "D1": D1,
                            "intercept": float(intercept), "r_value": float(r_value),
                            "p_value": float(p_value), "std_err": float(std_err),
                            "n_points": int(x.size)})
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
        """
        Compute alpha(q)=dτ/dq and f(α)=qα-τ(q) using a spline on τ(q).
        """
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
    def get_mfs(self, max_size=None, max_scales=80, min_points=6, min_box=2):
        df_mass = self.get_mass_table(max_size=max_size, max_scales=max_scales, min_box=min_box)
        df_fit = self.fit_tau_and_D1(df_mass, min_points=min_points, require_common_scales=True)
        df_spec = self.alpha_falpha_from_tau(df_fit, exclude_q1=True, spline_s=0)
        return df_mass, df_fit, df_spec

    # ------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------
    def plot(self, df_mass, df_fit, df_spec):
        fig, axs = plt.subplots(2, 2, figsize=(8, 6))

        # --- Heatmap for Mq
        df_mq = df_mass[df_mass["kind"] == "Mq"].copy()
        if not df_mq.empty:
            df_mq["log_value"] = np.log(df_mq["value"].astype(np.float64))
            pivot = df_mq.pivot_table(index="scale", columns="q", values="log_value", aggfunc="mean")
            vals = pivot.values
            if np.any(np.isfinite(vals)):
                vmin = np.nanpercentile(vals, 10)
                vmax = np.nanpercentile(vals, 90)
                sns.heatmap(pivot, ax=axs[0, 0], cmap="coolwarm", vmin=vmin, vmax=vmax, cbar=True)
                axs[0, 0].set_xlabel("q")
                axs[0, 0].set_ylabel("scale (box size ε)")
                axs[0, 0].set_title("Heatmap: log M(q, ε) vs scale and q")
                axs[0, 0].xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            else:
                axs[0, 0].set_title("Heatmap: (all NaN)")
        else:
            axs[0, 0].set_title("Heatmap: (no data)")

        # --- tau(q)
        if df_fit is not None and not df_fit.empty:
            sns.lineplot(data=df_fit, x="q", y="tau", ax=axs[0, 1])
            axs[0, 1].set_xlabel("q")
            axs[0, 1].set_ylabel(r"$\tau(q)$")
            axs[0, 1].set_title(r"$\tau(q)$ vs $q$")
            axs[0, 1].grid(True)
        else:
            axs[0, 1].set_title("tau(q): (no data)")

        # --- D(q)
        if df_fit is not None and not df_fit.empty:
            sns.lineplot(data=df_fit, x="q", y="Dq", ax=axs[1, 0])
            axs[1, 0].set_xlabel("q")
            axs[1, 0].set_ylabel(r"$D(q)$")
            axs[1, 0].set_title(r"$D(q)$ vs $q$ (with $D_1$ at q=1)")
            axs[1, 0].grid(True)
        else:
            axs[1, 0].set_title("D(q): (no data)")

        # --- f(α) vs α
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
    image_path = "../images/face.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)

    q_list = np.linspace(-5, 5, 51)

    mfs = CFA2DMFSBoxCounting(
        image=image,
        corp_type=-1,
        q_list=q_list,
        with_progress=True,
    )

    df_mass, df_fit, df_spec = mfs.get_mfs(max_scales=80, min_points=6)

    print("\n=== Fit results (head) ===")
    print(df_fit.head(10))

    row_d1 = df_fit[np.isclose(df_fit["q"], 1.0)]
    if not row_d1.empty:
        print("\n=== D1 (information dimension) ===")
        print(row_d1[["q", "D1", "Dq", "n_points", "r_value", "std_err"]])

    mfs.plot(df_mass, df_fit, df_spec)


if __name__ == "__main__":
    main()
