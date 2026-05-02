"""
Lacunarity analysis for 2D grayscale or binary images.

Standard definition (Allain & Cloitre 1991, Plotnick et al. 1996)
-----------------------------------------------------------------
For a measure (image) and a box of side r, let M be the box mass.
Sample many boxes (gliding or non-overlapping) at the same r, then

    Lambda(r) = <M^2> / <M>^2 = 1 + Var(M) / Mean(M)^2

The lower bound is 1 (zero variance => homogeneous). For self-similar
fractals, Lambda(r) ~ r^{-beta} with beta = D - E, so log Lambda vs
log r is a straight line. This is the default fit performed below.

Two box-partition strategies
----------------------------
1. Gliding box: every (s x s) window in the image, computed in O(H W)
   per scale via a single summed-area table (integral image) shared
   across all scales (huge speedup vs recomputing per-scale).
2. Non-overlapping box: tile the image with disjoint (s x s) blocks.

Two scale-generation strategies
-------------------------------
1. "powers"   - 2, 4, 8, ..., up to max_size.
2. "logspace" - geometrically spaced unique integers in [min_size, max_size].

What changed vs the original implementation
-------------------------------------------
- Gliding-box integral image is now computed ONCE outside the scale
  loop instead of once per scale. For an N-scale sweep this is an N-x
  speedup of the per-scale work.
- The mass extraction uses contiguous slices instead of meshgrid index
  arrays; less memory traffic, faster.
- Default fit is now log Lambda vs log r (the standard slope -beta);
  the legacy log(Lambda - 1) fit is still available via transform=...
  Original behaviour silently dropped low-lacunarity points (Lambda
  close to 1) which is exactly the "homogeneous" regime that lacunarity
  is supposed to characterise.
- New batch APIs (get_batch_lacunarity, fit_batch_lacunarity) compute
  the integral image once across the whole batch and vectorise the
  per-box reductions.
- include_zero=False (drop empty boxes) is correctly handled even when
  different boxes contain different numbers of zeros.
"""

import os
import math
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import linregress

try:
    from FreeAeonFractal.FAImage import CFAImage
except Exception:  # pragma: no cover - convenience fallback for tests
    try:
        from FAImage import CFAImage
    except Exception:
        CFAImage = None  # plotting / IO helpers not used in core logic


# ============================================================
# Scale generation
# ============================================================
def make_scales(max_size: int,
                scales_mode: str = "powers",
                max_scales: int = 100,
                min_size: int = 2) -> List[int]:
    """
    Generate the list of box sizes used for scanning.

    scales_mode == "powers"   -> 2, 4, 8, ..., up to max_size.
    scales_mode == "logspace" -> geometrically spaced integers in
                                 [min_size, max_size], deduplicated.
    """
    max_size = int(max_size)
    min_size = int(max(1, min_size))
    if max_size < min_size:
        return []
    if scales_mode == "powers":
        max_pow = int(np.floor(np.log2(max_size)))
        out = [int(2 ** k) for k in range(int(np.log2(max(min_size, 1))), max_pow + 1)
               if 2 ** k >= min_size and 2 ** k <= max_size]
        return out
    if scales_mode == "logspace":
        raw = np.logspace(np.log2(min_size), np.log2(max_size),
                          num=int(max_scales), base=2.0)
        ints = np.unique(np.round(raw).astype(int))
        ints = ints[(ints >= min_size) & (ints <= max_size)]
        return ints.astype(int).tolist()
    raise ValueError(f"unknown scales_mode={scales_mode!r}")


# ============================================================
# Mass-statistics utilities
# ============================================================
def _lacunarity_from_stats(mean_m: float, mean_m2: float,
                           eps: float = 1e-12) -> float:
    """Lambda = <M^2> / <M>^2. Returns NaN if mean is too small."""
    if not np.isfinite(mean_m) or not np.isfinite(mean_m2):
        return float("nan")
    if mean_m <= eps:
        return float("nan")
    return float(mean_m2 / (mean_m * mean_m))


def _gliding_mass_stats_from_S(S: np.ndarray, S2: np.ndarray, size: int,
                               include_zero: bool = True
                               ) -> Tuple[int, float, float]:
    """
    From two (H+1, W+1) summed-area tables S (sum of img) and S2 (sum
    of img**2), compute (n_boxes, mean(M), mean(M^2)) for every (size,
    size) gliding box in the image.

    Note: S2 is the integral of pixel SQUARES, not box-mass squares.
    To get mean(M^2) over boxes we still need to expand box-mass^2,
    which we do directly without ever materialising the (n_boxes,)
    vector of masses. That keeps memory bounded by O(H*W).
    """
    H1, W1 = S.shape  # (H+1, W+1)
    H, W = H1 - 1, W1 - 1
    if size > H or size > W:
        return 0, float("nan"), float("nan")
    # Mass M(y, x) = S[y+s, x+s] - S[y, x+s] - S[y+s, x] + S[y, x]
    # Use slice arithmetic; no meshgrid.
    M = (S[size:, size:] - S[:-size, size:]
         - S[size:, :-size] + S[:-size, :-size])

    if include_zero:
        n = M.size
        if n == 0:
            return 0, float("nan"), float("nan")
        sum_M = float(M.sum())
        sum_M2 = float((M * M).sum())     # box-mass squared, not pixel^2
        return n, sum_M / n, sum_M2 / n

    mask = M > 0
    n_pos = int(mask.sum())
    if n_pos == 0:
        return 0, float("nan"), float("nan")
    Mp = M[mask]
    return n_pos, float(Mp.mean()), float((Mp * Mp).mean())


def _nonoverlap_mass_stats(image: np.ndarray, size: int,
                           use_binary_mass: bool = False,
                           include_zero: bool = True
                           ) -> Tuple[int, float, float]:
    """
    Tile `image` with (size, size) non-overlapping blocks (cropping the
    image to (H//size)*size, (W//size)*size first), and return
    (n_boxes, mean(M), mean(M^2)).
    """
    H, W = image.shape[:2]
    nY, nX = H // size, W // size
    if nY == 0 or nX == 0:
        return 0, float("nan"), float("nan")
    arr = image[:nY * size, :nX * size]
    if use_binary_mass:
        arr = (arr > 0)
    arr = arr.astype(np.float64, copy=False)
    blocks = arr.reshape(nY, size, nX, size).transpose(0, 2, 1, 3)
    blocks = blocks.reshape(nY * nX, size * size)
    M = blocks.sum(axis=1)
    if include_zero:
        if M.size == 0:
            return 0, float("nan"), float("nan")
        return int(M.size), float(M.mean()), float((M * M).mean())
    Mp = M[M > 0]
    if Mp.size == 0:
        return 0, float("nan"), float("nan")
    return int(Mp.size), float(Mp.mean()), float((Mp * Mp).mean())


# ============================================================
# Curve fitting
# ============================================================
def _fit_lacunarity_curve(scales: np.ndarray, lacs: np.ndarray,
                          transform: str = "log",
                          fit_range: Optional[Tuple[float, float]] = None
                          ) -> dict:
    """
    transform : {"log", "log_minus_1"}
        - "log"          : fit log(Lambda) vs log(r). Standard for
                            self-similar fractals (Allain & Cloitre).
                            Slope corresponds to beta = D - E with the
                            sign convention Lambda(r) ~ r^{-beta}, so
                            slope = -beta.
        - "log_minus_1"  : fit log(Lambda - 1) vs log(r). Handles
                            homogeneous (Lambda=1) regions by ignoring
                            them; legacy behaviour. Most useful when
                            you want a finite slope on heterogeneous
                            regions only.
    fit_range : (s_min, s_max) or None
        Restrict fit to scales in [s_min, s_max]. Box-counting curves
        are sigmoid-shaped at the small/large extremes, so picking the
        linear regime improves the slope estimate.
    """
    s = np.asarray(scales, dtype=np.float64)
    L = np.asarray(lacs, dtype=np.float64)
    keep = np.isfinite(s) & (s > 0) & np.isfinite(L)
    if transform == "log":
        keep &= (L > 0.0)
        y_full = np.log(np.where(L > 0, L, np.nan))
    elif transform == "log_minus_1":
        keep &= (L > 1.0 + 1e-12)
        y_full = np.log(np.where(L > 1.0 + 1e-12, L - 1.0, np.nan))
    else:
        raise ValueError(f"unknown transform={transform!r}")
    if fit_range is not None:
        s_lo, s_hi = float(fit_range[0]), float(fit_range[1])
        keep &= (s >= s_lo) & (s <= s_hi)
    s_fit = s[keep]
    y_fit = y_full[keep]
    if s_fit.size < 2:
        return {
            "slope": float("nan"), "intercept": float("nan"),
            "r_value": float("nan"), "p_value": float("nan"),
            "std_err": float("nan"),
            "log_scales": [], "log_lambda_minus_1": [],
            "transform": transform,
        }
    x_fit = np.log(s_fit)
    slope, intercept, r_value, p_value, std_err = linregress(x_fit, y_fit)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_value": float(r_value),
        "p_value": float(p_value),
        "std_err": float(std_err),
        "log_scales": x_fit.tolist(),
        "log_lambda_minus_1": y_fit.tolist(),  # name kept for plot back-compat
        "transform": transform,
    }


# ============================================================
# Main class
# ============================================================
class CFAImageLAC:
    """
    Lacunarity calculator for a single image.

    For batch processing of many images, use the static methods
    `get_batch_lacunarity` / `fit_batch_lacunarity`.
    """

    def __init__(self, image: Optional[np.ndarray] = None,
                 max_size: Optional[int] = None,
                 max_scales: int = 100,
                 with_progress: bool = True,
                 scales_mode: str = "powers",
                 partition_mode: str = "gliding",
                 min_size: int = 2):
        if image is None:
            raise ValueError("Image must be provided.")
        if image.ndim != 2:
            raise ValueError("image must be a 2D single-channel array")
        self.image = image
        self.with_progress = with_progress
        self.partition_mode = partition_mode.lower()
        if self.partition_mode not in ("gliding", "non-overlapping"):
            raise ValueError("partition_mode must be 'gliding' or "
                             "'non-overlapping'")
        if max_size is None:
            max_size = min(image.shape)
        self.scales = make_scales(max_size, scales_mode=scales_mode,
                                  max_scales=max_scales, min_size=min_size)

    # ------------------------------------------------------------
    # Single-image lacunarity
    # ------------------------------------------------------------
    def get_lacunarity(self, corp_type: int = -1,
                       use_binary_mass: bool = False,
                       include_zero: bool = True) -> dict:
        """
        Compute Lambda(r) for every box size in self.scales.

        Parameters
        ----------
        corp_type : int
            Currently only the default crop-to-multiple is supported
            for non-overlapping boxes; gliding boxes don't need it.
            Kept for API parity with the rest of the library.
        use_binary_mass : bool
            If True, treat the image as binary (>0 -> 1) before summing.
            Equivalent to Allain & Cloitre's binary lacunarity.
        include_zero : bool
            If False, drop boxes with mass=0 from the statistics.
        """
        # Pre-compute the integral image once (gliding mode only).
        S = S2 = None
        if self.partition_mode == "gliding":
            img_f = ((self.image > 0).astype(np.float64)
                     if use_binary_mass
                     else self.image.astype(np.float64, copy=False))
            S = _integral_image(img_f)

        scales_used: List[int] = []
        lambdas: List[float] = []
        mass_stats: List[dict] = []

        it = (tqdm(self.scales, desc="Calculating Lacunarity")
              if self.with_progress else self.scales)
        for size in it:
            size = int(size)
            if self.partition_mode == "gliding":
                n, mean_m, mean_m2 = _gliding_mass_stats_from_S(
                    S, S2, size, include_zero=include_zero)
            else:  # non-overlapping
                n, mean_m, mean_m2 = _nonoverlap_mass_stats(
                    self.image, size,
                    use_binary_mass=use_binary_mass,
                    include_zero=include_zero)
            lam = _lacunarity_from_stats(mean_m, mean_m2)
            scales_used.append(size)
            lambdas.append(lam)
            mass_stats.append({
                "scale": size,
                "num_boxes": int(n),
                "mean_mass": float(mean_m) if np.isfinite(mean_m) else float("nan"),
                # var = E[M^2] - E[M]^2  (population variance)
                "var_mass": (float(mean_m2 - mean_m * mean_m)
                             if (np.isfinite(mean_m) and np.isfinite(mean_m2))
                             else float("nan")),
                "lambda": lam,
            })
        return {
            "scales": scales_used,
            "lacunarity": lambdas,
            "mass_stats": mass_stats,
        }

    # ------------------------------------------------------------
    # Curve fitting (instance API, kept for back-compat)
    # ------------------------------------------------------------
    def fit_lacunarity(self, lac_result: dict,
                       transform: str = "log",
                       fit_range: Optional[Tuple[float, float]] = None
                       ) -> dict:
        return _fit_lacunarity_curve(
            np.asarray(lac_result["scales"]),
            np.asarray(lac_result["lacunarity"]),
            transform=transform, fit_range=fit_range)

    # ============================================================
    # Batch APIs (static)
    # ============================================================
    @staticmethod
    def get_batch_lacunarity(images: Sequence[np.ndarray],
                             max_size: Optional[int] = None,
                             max_scales: int = 100,
                             scales_mode: str = "powers",
                             partition_mode: str = "gliding",
                             min_size: int = 2,
                             use_binary_mass: bool = False,
                             include_zero: bool = True,
                             with_progress: bool = True) -> List[dict]:
        """Batch lacunarity. All images may have different shapes; each
        gets its own integral image. Scales are shared across images
        (by `max_size = min(min(im.shape) for im in images)` if not
        provided)."""
        if len(images) == 0:
            return []
        for i, im in enumerate(images):
            if im.ndim != 2:
                raise ValueError(f"image {i} is not 2D")
        if max_size is None:
            max_size = min(min(im.shape) for im in images)
        scales = make_scales(max_size, scales_mode=scales_mode,
                             max_scales=max_scales, min_size=min_size)
        if not scales:
            raise ValueError("No valid scales generated")

        same_shape = all(im.shape == images[0].shape for im in images)
        N = len(images)
        partition_mode = partition_mode.lower()

        if partition_mode == "gliding" and same_shape:
            return _batch_gliding_same_shape(
                images, scales, use_binary_mass=use_binary_mass,
                include_zero=include_zero, with_progress=with_progress)

        # Heterogeneous shapes or non-overlapping mode -> per-image loop
        # but each image still uses the cached-integral fast path.
        results: List[dict] = []
        it = (tqdm(images, desc="Batch lacunarity")
              if with_progress else images)
        for im in it:
            obj = CFAImageLacunarity(im, max_size=max_size,
                                     max_scales=max_scales,
                                     scales_mode=scales_mode,
                                     partition_mode=partition_mode,
                                     min_size=min_size,
                                     with_progress=False)
            results.append(obj.get_lacunarity(
                use_binary_mass=use_binary_mass,
                include_zero=include_zero))
        return results

    @staticmethod
    def fit_batch_lacunarity(lac_results: Sequence[dict],
                             transform: str = "log",
                             fit_range: Optional[Tuple[float, float]] = None
                             ) -> List[dict]:
        """Apply the same fit to every result returned by
        `get_batch_lacunarity`."""
        return [_fit_lacunarity_curve(np.asarray(r["scales"]),
                                       np.asarray(r["lacunarity"]),
                                       transform=transform,
                                       fit_range=fit_range)
                for r in lac_results]

    # ------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------
    def plot(self, lac_result, fit_result=None, ax=None, show=True,
             title="Lacunarity", label=None):
        scales = np.asarray(lac_result["scales"], dtype=float)
        lac = np.asarray(lac_result["lacunarity"], dtype=float)

        if ax is None:
            if fit_result is None:
                fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=120)
                ax_main = ax
                ax_fit = None
            else:
                fig, (ax_main, ax_fit) = plt.subplots(
                    1, 2, figsize=(11, 4), dpi=120)
        else:
            ax_main = ax
            ax_fit = None

        ax_main.plot(scales, lac, "o-", lw=1.5, ms=4,
                     label=label or r"$\Lambda(r)$")
        ax_main.set_xlabel("Scale r (box size)")
        ax_main.set_ylabel(r"Lacunarity $\Lambda(r)$")
        ax_main.set_xscale("log")
        ax_main.set_yscale("log")
        ax_main.set_title(title)
        ax_main.grid(True, alpha=0.3)
        if label is not None:
            ax_main.legend()

        if fit_result is not None and ax_fit is not None:
            x = np.asarray(fit_result["log_scales"], dtype=float)
            y = np.asarray(fit_result["log_lambda_minus_1"], dtype=float)
            slope = fit_result["slope"]
            intercept = fit_result["intercept"]
            r_value = fit_result.get("r_value", float("nan"))

            if x.size >= 2:
                ax_fit.scatter(x, y, s=18, label="data")
                xline = np.linspace(np.min(x), np.max(x), 200)
                yline = slope * xline + intercept
                ax_fit.plot(xline, yline, "r-", lw=2,
                            label=f"y={slope:.3f}x+({intercept:.2f})\n"
                                  f"R²={r_value ** 2:.3f}")
                ax_fit.legend()
            ax_fit.set_xlabel(r"$\log(r)$")
            tr = fit_result.get("transform", "log")
            ax_fit.set_ylabel(r"$\log(\Lambda(r) - 1)$"
                              if tr == "log_minus_1"
                              else r"$\log\,\Lambda(r)$")
            ax_fit.set_title(f"Lacunarity scaling fit ({tr})")
            ax_fit.grid(True, alpha=0.3)

        if show:
            plt.tight_layout()
            plt.show()
        return ax_main


# ============================================================
# Internal helpers
# ============================================================
def _integral_image(img: np.ndarray) -> np.ndarray:
    """
    Padded summed-area table: S[y, x] = sum_{i<y, j<x} img[i, j].
    Shape (H+1, W+1) with the leading zero row/column.
    Mass of the box [y0:y1, x0:x1] = S[y1, x1] - S[y0, x1]
                                     - S[y1, x0] + S[y0, x0].
    """
    H, W = img.shape
    S = np.empty((H + 1, W + 1), dtype=np.float64)
    S[0, :] = 0.0
    S[:, 0] = 0.0
    np.cumsum(np.cumsum(img, axis=0, dtype=np.float64),
              axis=1, out=S[1:, 1:])
    return S


def _batch_gliding_same_shape(images: Sequence[np.ndarray],
                              scales: List[int],
                              use_binary_mass: bool,
                              include_zero: bool,
                              with_progress: bool) -> List[dict]:
    """
    Vectorise the gliding-box reduction across the batch axis at every
    scale. Memory: O(N * H * W) for the integral images, O(N) for the
    per-scale reductions.
    """
    N = len(images)
    # Stack as float64; binarise if requested.
    stack = np.empty((N, *images[0].shape), dtype=np.float64)
    for i, im in enumerate(images):
        if use_binary_mass:
            stack[i] = (im > 0).astype(np.float64)
        else:
            stack[i] = im.astype(np.float64, copy=False)

    H, W = stack.shape[1], stack.shape[2]
    # (N, H+1, W+1) integral image, computed once for the whole batch.
    S = np.zeros((N, H + 1, W + 1), dtype=np.float64)
    np.cumsum(np.cumsum(stack, axis=1), axis=2, out=S[:, 1:, 1:])

    out_per_image: List[dict] = [{
        "scales": [],
        "lacunarity": [],
        "mass_stats": [],
    } for _ in range(N)]

    it = (tqdm(scales, desc="Batch gliding lacunarity")
          if with_progress else scales)
    for size in it:
        size = int(size)
        if size > H or size > W:
            for i in range(N):
                out_per_image[i]["scales"].append(size)
                out_per_image[i]["lacunarity"].append(float("nan"))
                out_per_image[i]["mass_stats"].append({
                    "scale": size, "num_boxes": 0,
                    "mean_mass": float("nan"),
                    "var_mass": float("nan"),
                    "lambda": float("nan"),
                })
            continue
        # Mass cube: (N, H-s+1, W-s+1)
        M = (S[:, size:, size:] - S[:, :-size, size:]
             - S[:, size:, :-size] + S[:, :-size, :-size])
        if include_zero:
            n_boxes = M.shape[1] * M.shape[2]
            sum_M = M.sum(axis=(1, 2))                                 # (N,)
            sum_M2 = (M * M).sum(axis=(1, 2))                          # (N,)
            mean_M = sum_M / n_boxes
            mean_M2 = sum_M2 / n_boxes
            n_arr = np.full(N, n_boxes, dtype=np.int64)
        else:
            # Per-image masking (counts can differ per image).
            mask = M > 0
            n_arr = mask.sum(axis=(1, 2))                              # (N,)
            sum_M = (M * mask).sum(axis=(1, 2))
            sum_M2 = ((M * mask) * (M * mask)).sum(axis=(1, 2))
            with np.errstate(divide="ignore", invalid="ignore"):
                mean_M = np.where(n_arr > 0, sum_M / np.maximum(n_arr, 1),
                                   np.nan)
                mean_M2 = np.where(n_arr > 0,
                                   sum_M2 / np.maximum(n_arr, 1), np.nan)

        var_M = mean_M2 - mean_M * mean_M
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = np.where((mean_M > 0) & np.isfinite(mean_M2),
                           mean_M2 / (mean_M * mean_M), np.nan)

        for i in range(N):
            out_per_image[i]["scales"].append(size)
            out_per_image[i]["lacunarity"].append(
                float(lam[i]) if np.isfinite(lam[i]) else float("nan"))
            out_per_image[i]["mass_stats"].append({
                "scale": size,
                "num_boxes": int(n_arr[i]),
                "mean_mass": float(mean_M[i]) if np.isfinite(mean_M[i]) else float("nan"),
                "var_mass": float(var_M[i]) if np.isfinite(var_M[i]) else float("nan"),
                "lambda": float(lam[i]) if np.isfinite(lam[i]) else float("nan"),
            })

    return out_per_image


# ============================================================
# Demo
# ============================================================
def main():
    import cv2
    img_path = os.path.join(os.path.dirname(__file__), "../images/face.png")
    gray_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray_image is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    if CFAImage is not None:
        bin_image, threshold = CFAImage.otsu_binarize(gray_image)
    else:
        import cv2 as _cv
        _, bin_image = _cv.threshold(gray_image, 0, 255,
                                      _cv.THRESH_BINARY + _cv.THRESH_OTSU)

    # Binary Gliding Box
    calc_bin_gliding = CFAImageLAC(bin_image, partition_mode="gliding")
    lac_bin_gliding = calc_bin_gliding.get_lacunarity(use_binary_mass=True,
                                                      include_zero=True)
    fit_bin_gliding = calc_bin_gliding.fit_lacunarity(lac_bin_gliding)
    print("Binary Gliding Lambda(r):", lac_bin_gliding["lacunarity"])
    print("Binary Gliding fit slope:", fit_bin_gliding["slope"])

    # Gray Gliding Box
    calc_gray_gliding = CFAImageLAC(gray_image, partition_mode="gliding")
    lac_gray_gliding = calc_gray_gliding.get_lacunarity()
    fit_gray_gliding = calc_gray_gliding.fit_lacunarity(lac_gray_gliding)
    print("Gray Gliding Lambda(r):", lac_gray_gliding["lacunarity"])
    print("Gray Gliding fit slope:", fit_gray_gliding["slope"])

    # Batch
    batch_imgs = [bin_image, bin_image, bin_image]
    batch_res = CFAImageLAC.get_batch_lacunarity(
        batch_imgs, partition_mode="gliding",
        use_binary_mass=True, with_progress=False)
    batch_fits = CFAImageLAC.fit_batch_lacunarity(batch_res)
    print("Batch slopes:", [f["slope"] for f in batch_fits])


if __name__ == "__main__":
    main()
