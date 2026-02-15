import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.stats import linregress
from FreeAeonFractal.FAImage import CFAImage
import cv2
import numpy as np

class CFAImageLacunarity(object):
    """
    Lacunarity for 2D images using non-overlapping box partition.

    image: input image (single channel, 2D)
    max_size: maximum box size for partitioning
    max_scales: number of scales sampled (logspace base 2)
    with_progress: show tqdm
    """
    def __init__(self, image=None, max_size=None, max_scales=100, with_progress=True):
        self.m_image = image
        self.m_with_progress = with_progress

        if max_size is None:
            max_size = min(image.shape) // 1

        scales = np.logspace(1, np.log2(max_size), num=max_scales, base=2, dtype=int)
        scales = np.unique(scales)
        self.m_scales = [int(s) for s in scales if s > 0]

    @staticmethod
    def _safe_lacunarity_from_masses(masses, eps=1e-12):
        """
        masses: 1D array of box masses (include zeros if you want)
        Λ = E[m^2]/(E[m])^2
        """
        m = np.asarray(masses, dtype=np.float64)
        if m.size == 0:
            return np.nan

        mean_m = m.mean()
        if mean_m <= eps:
            return np.nan

        mean_m2 = (m * m).mean()
        return float(mean_m2 / (mean_m * mean_m))

    def get_lacunarity(self, corp_type=-1, use_binary_mass=False, include_zero=True):
        """
        Compute lacunarity for each scale.

        corp_type: (-1 crop, 0 no processing, 1 padding) forwarded to get_boxes_from_image
        use_binary_mass:
            - True: mass = count of (box>0)
            - False: mass = sum of pixel values in box
        include_zero:
            - True: include zero-mass boxes (recommended)
            - False: drop zero-mass boxes
        """
        scale_list = []
        lambda_list = []
        mass_stats = []  # optional diagnostics per scale

        iterator = tqdm(self.m_scales, desc="Calculating Lacunarity") if self.m_with_progress else self.m_scales

        for size in iterator:
            block_size = (size, size)
            boxes, _ = CFAImage.get_boxes_from_image(self.m_image, block_size, corp_type=corp_type)

            if use_binary_mass:
                # count nonzero pixels per box
                axis = tuple(range(1, boxes.ndim))
                masses = np.sum(boxes > 0, axis=axis).astype(np.float64)
            else:
                # sum intensity per box
                axis = tuple(range(1, boxes.ndim))
                masses = np.sum(boxes, axis=axis).astype(np.float64)

            if not include_zero:
                masses = masses[masses > 0]

            lam = self._safe_lacunarity_from_masses(masses)

            scale_list.append(size)
            lambda_list.append(lam)
            if masses.size > 0:
                mass_stats.append({
                    "scale": int(size),
                    "num_boxes": int(masses.size),
                    "mean_mass": float(np.mean(masses)) if masses.size else np.nan,
                    "var_mass": float(np.var(masses)) if masses.size else np.nan,
                    "lambda": lam
                })
            else:
                mass_stats.append({
                    "scale": int(size),
                    "num_boxes": 0,
                    "mean_mass": np.nan,
                    "var_mass": np.nan,
                    "lambda": np.nan
                })

        return {
            "scales": scale_list,
            "lacunarity": lambda_list,
            "mass_stats": mass_stats,
        }

    def fit_lacunarity(self, lac_result, min_valid_lambda=1.0 + 1e-12):
        """
        Fit log(Λ-1) vs log(1/r).

        lac_result: dict returned by get_lacunarity()
        """
        scales = np.asarray(lac_result["scales"], dtype=np.float64)
        lam = np.asarray(lac_result["lacunarity"], dtype=np.float64)

        # valid: finite, and Λ>1 so that log(Λ-1) defined
        mask = np.isfinite(scales) & (scales > 0) & np.isfinite(lam) & (lam > min_valid_lambda)
        x = np.log(1.0 / scales[mask])
        y = np.log(lam[mask] - 1.0)

        if x.size < 2:
            return {
                "slope": np.nan,
                "intercept": np.nan,
                "r_value": np.nan,
                "p_value": np.nan,
                "std_err": np.nan,
                "log_inv_scales": x.tolist(),
                "log_lambda_minus_1": y.tolist()
            }

        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_value": float(r_value),
            "p_value": float(p_value),
            "std_err": float(std_err),
            "log_inv_scales": x.tolist(),
            "log_lambda_minus_1": y.tolist()
        }

    def plot(self, lac_result, fit_result=None, ax=None, show=True,
                        title="Lacunarity", label=None):
        """
        lac_result: dict returned by get_lacunarity()
        fit_result: dict returned by fit_lacunarity() (optional)
        ax: optional, matplotlib Axes. If None, create new figure.
        """
        scales = np.asarray(lac_result["scales"], dtype=float)
        lac = np.asarray(lac_result["lacunarity"], dtype=float)

        if ax is None:
            if fit_result is None:
                fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=120)
                axes = [ax]
            else:
                fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=120)
                ax = axes[0]
        else:
            axes = [ax]

        # (1) Λ(r) vs r
        ax.plot(scales, lac, "o-", lw=1.5, ms=4, label=label or r"$\Lambda(r)$")
        ax.set_xlabel("Scale r (box size)")
        ax.set_ylabel(r"Lacunarity $\Lambda(r)$")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if label is not None:
            ax.legend()

        # (2) log-log fit: log(Λ-1) vs log(1/r)
        if fit_result is not None:
            ax2 = axes[1]
            x = np.asarray(fit_result["log_inv_scales"], dtype=float)
            y = np.asarray(fit_result["log_lambda_minus_1"], dtype=float)

            slope = fit_result["slope"]
            intercept = fit_result["intercept"]
            r_value = fit_result.get("r_value", np.nan)

            ax2.scatter(x, y, s=18, label="data")
            xline = np.linspace(np.min(x), np.max(x), 200)
            yline = slope * xline + intercept
            ax2.plot(xline, yline, "r-", lw=2,
                     label=f"y={slope:.2f}x+({intercept:.2f})\n"
                           f"r={r_value:.2f},R²={r_value**2:.2f}")
            ax2.set_xlabel(r"$\log(1/r)$")
            ax2.set_ylabel(r"$\log(\Lambda(r)-1)$")
            ax2.set_title("Lacunarity scaling fit")
            ax2.grid(True, alpha=0.3)
            ax2.legend()

        if show:
            plt.tight_layout()
            plt.show()

        return ax

def main():
    raw_image = cv2.imread("../images/fractal.png", cv2.IMREAD_GRAYSCALE)

    # for binary images
    bin_image = (raw_image < 64).astype(np.uint8)
    lac_bin = CFAImageLacunarity(bin_image).get_lacunarity(
        corp_type=-1, use_binary_mass=True, include_zero=True
    )
    fit_bin = CFAImageLacunarity(bin_image).fit_lacunarity(lac_bin)

    print("Lacunarity (per scale):", lac_bin["lacunarity"])
    print("Fit slope:", fit_bin["slope"], "R:", fit_bin["r_value"])

    # for gray images
    
    lac_gray = CFAImageLacunarity(raw_image).get_lacunarity(
        corp_type=-1, use_binary_mass=False, include_zero=True
    )
    fit_gray = CFAImageLacunarity(raw_image).fit_lacunarity(lac_gray)
    print("Gray lacunarity:", lac_gray["lacunarity"])
    print("Fit slope:", fit_gray["slope"], "R:", fit_gray["r_value"])
    CFAImageLacunarity(raw_image).plot(lac_gray,fit_gray)

if __name__ == "__main__":
    main()

