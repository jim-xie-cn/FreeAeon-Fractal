"""
Implements lacunarity analysis for 2D grayscale or binary images.

Two box partition strategies:
1. Gliding box partition — sliding windows that overlap, computed efficiently using summed-area tables (integral images).
2. Non-overlapping box partition — fixed-size blocks tiled over the image without overlap.

Two box sizes scales strategies:
1. powers
2. logspace 
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.stats import linregress
from FreeAeonFractal.FAImage import CFAImage
import cv2

class CFAImageLacunarity:
    """
    Lacunarity analysis for 2D images using non-overlapping or gliding box partition.
    """

    def __init__(self, image=None, max_size=None, max_scales=100, with_progress=True, 
                 scales_mode="powers", partition_mode="gliding"):

        if image is None:
            raise ValueError("Image must be provided.")

        self.image = image
        self.with_progress = with_progress
        self.partition_mode = partition_mode.lower()

        if max_size is None:
            max_size = min(image.shape)

        # Generate log-spaced or powers-of-two scales
        if scales_mode == "powers":
            max_pow = int(np.log2(max_size))
            self.scales = list(2 ** np.arange(1, max_pow + 1))
        elif scales_mode == "logspace":
            scales = np.logspace(0, np.log2(max_size), num=max_scales, base=2)
            #scales = np.logspace(1, np.log2(max_size), num=max_scales, base=2)
            scales = np.unique(np.round(scales).astype(int))
            scales = scales[scales <= max_size] 
            self.scales = [int(s) for s in scales if s > 0]

    @staticmethod
    def _safe_lacunarity_from_masses(masses, eps=1e-12):
        """Compute lacunarity from box masses."""
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
        Compute lacunarity for each scale using either:
        - non-overlapping boxes
        - gliding boxes
        """
        scale_list, lambda_list, mass_stats = [], [], []
        iterator = tqdm(self.scales, desc="Calculating Lacunarity") if self.with_progress else self.scales

        for size in iterator:
            if self.partition_mode == "non-overlapping":
                block_size = (size, size)
                boxes, _ = CFAImage.get_boxes_from_image(self.image, block_size, corp_type=corp_type)
                axis = tuple(range(1, boxes.ndim))
                masses = np.sum((boxes > 0) if use_binary_mass else boxes, axis=axis).astype(np.float64)

            elif self.partition_mode == "gliding":
                img = (self.image > 0).astype(np.float64) if use_binary_mass else self.image.astype(np.float64)
                S = np.pad(np.cumsum(np.cumsum(img, axis=0), axis=1), ((1, 0), (1, 0)), mode="constant")
                y_max, x_max = img.shape
                y1 = np.arange(size, y_max + 1)
                x1 = np.arange(size, x_max + 1)
                Y1, X1 = np.meshgrid(y1, x1, indexing="ij")
                Y0, X0 = Y1 - size, X1 - size
    
                masses = S[Y1, X1] - S[Y0, X1] - S[Y1, X0] + S[Y0, X0]
                masses = masses.ravel().astype(np.float64)

            if not include_zero:
                masses = masses[masses > 0]
            
            if masses.size == 0:
                print(f"[Warning] No non-zero masses at scale {size}")

            lambda_val = self._safe_lacunarity_from_masses(masses)
            scale_list.append(size)
            lambda_list.append(lambda_val)
            mass_stats.append({
                "scale": int(size),
                "num_boxes": int(masses.size),
                "mean_mass": float(np.mean(masses)) if masses.size else np.nan,
                "var_mass": float(np.var(masses)) if masses.size else np.nan,
                "lambda": lambda_val
            })

        return {
            "scales": scale_list,
            "lacunarity": lambda_list,
            "mass_stats": mass_stats
        }

    def fit_lacunarity(self, lac_result, min_valid_lambda=1.0 + 1e-12):
        """Fit log(Λ - 1) vs log(r) using only valid values."""
        scales = np.asarray(lac_result["scales"], dtype=np.float64)
        lac_vals = np.asarray(lac_result["lacunarity"], dtype=np.float64)

        mask = np.isfinite(scales) & (scales > 0) & np.isfinite(lac_vals) & (lac_vals > min_valid_lambda)
        x = np.log(scales[mask])
        y = np.log(lac_vals[mask] - 1.0)

        if x.size < 2:
            return {
                "slope": np.nan,
                "intercept": np.nan,
                "r_value": np.nan,
                "p_value": np.nan,
                "std_err": np.nan,
                "log_scales": x.tolist(),
                "log_lambda_minus_1": y.tolist()
            }

        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_value": float(r_value),
            "p_value": float(p_value),
            "std_err": float(std_err),
            "log_scales": x.tolist(),
            "log_lambda_minus_1": y.tolist()
        }

    def plot(self, lac_result, fit_result=None, ax=None, show=True, title="Lacunarity", label=None):
        """Plot lacunarity curves and optional log-log fit."""
        scales = np.asarray(lac_result["scales"], dtype=float)
        lac = np.asarray(lac_result["lacunarity"], dtype=float)

        if ax is None:
            if fit_result is None:
                fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=120)
                ax_main = ax
            else:
                fig, (ax_main, ax_fit) = plt.subplots(1, 2, figsize=(11, 4), dpi=120)
        else:
            ax_main = ax
            ax_fit = None

        ax_main.plot(scales, lac, "o-", lw=1.5, ms=4, label=label or r"$\Lambda(r)$")
        ax_main.set_xlabel("Scale r (box size)")
        ax_main.set_ylabel(r"Lacunarity $\Lambda(r)$")
        ax_main.set_title(title)
        ax_main.grid(True, alpha=0.3)
        if label is not None:
            ax_main.legend()

        if fit_result is not None and ax_fit is not None:
            x = np.asarray(fit_result["log_scales"], dtype=float)
            y = np.asarray(fit_result["log_lambda_minus_1"], dtype=float)
            slope = fit_result["slope"]
            intercept = fit_result["intercept"]
            r_value = fit_result.get("r_value", np.nan)

            ax_fit.scatter(x, y, s=18, label="data")
            xline = np.linspace(np.min(x), np.max(x), 200)
            yline = slope * xline + intercept
            ax_fit.plot(
                xline, yline, "r-", lw=2,
                label=f"y={slope:.2f}x+({intercept:.2f})\n"
                      f"r={r_value:.2f}, R²={r_value**2:.2f}"
            )
            ax_fit.set_xlabel(r"$\log(r)$")
            ax_fit.set_ylabel(r"$\log(\Lambda(r)-1)$")
            ax_fit.set_title("Lacunarity scaling fit")
            ax_fit.grid(True, alpha=0.3)
            ax_fit.legend()

        if show:
            plt.tight_layout()
            plt.show()

        return ax_main


def main():
    img_path = os.path.join(os.path.dirname(__file__), "../images/face.png")
    gray_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray_image is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    # Binary Gliding Box
    bin_image,threshold = CFAImage.otsu_binarize(gray_image)
    calc_bin_gliding = CFAImageLacunarity(bin_image, partition_mode="gliding")
    lac_bin_gliding = calc_bin_gliding.get_lacunarity(use_binary_mass=True, include_zero=True)
    fit_bin_gliding = calc_bin_gliding.fit_lacunarity(lac_bin_gliding)
    print("Binary Gliding Lacunarity:", lac_bin_gliding["lacunarity"], "Slope:", fit_bin_gliding["slope"])

    # Gray Gliding Box
    calc_gray_gliding = CFAImageLacunarity(gray_image, partition_mode="gliding")
    lac_gray_gliding = calc_gray_gliding.get_lacunarity(use_binary_mass=False, include_zero=True)
    fit_gray_gliding = calc_gray_gliding.fit_lacunarity(lac_gray_gliding)
    print("Gray Gliding Lacunarity:", lac_gray_gliding["lacunarity"], "Slope:", fit_gray_gliding["slope"])

    # Gray Non-overlapping
    calc_gray_nonoverlap = CFAImageLacunarity(gray_image, partition_mode="non-overlapping")
    lac_gray_nonoverlap = calc_gray_nonoverlap.get_lacunarity(use_binary_mass=False, include_zero=True)
    fit_gray_nonoverlap = calc_gray_nonoverlap.fit_lacunarity(lac_gray_nonoverlap)
    print("Gray Non-overlap Lacunarity:", lac_gray_nonoverlap["lacunarity"], "Slope:", fit_gray_nonoverlap["slope"])

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    calc_bin_gliding.plot(lac_bin_gliding, fit_bin_gliding, ax=ax, show=False, label="Gliding Box (Bin)")
    calc_gray_gliding.plot(lac_gray_gliding, fit_gray_gliding, ax=ax, show=False, label="Gliding Box (Gray)")
    calc_gray_nonoverlap.plot(lac_gray_nonoverlap, fit_gray_nonoverlap, ax=ax, show=False, label="Non-overlapping (Gray)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

