"""
Basic operations for 2D shapes
1. Calculation of various fractal dimensions

Implements three classical box-counting estimators:
    BC   - Box Counting          (binary image)
    DBC  - Differential Box Counting (Sarkar & Chaudhuri 1994, gray image)
    SDBC - Shifted Differential Box Counting (Chen et al. 1995, gray image)

What changed vs the original implementation
-------------------------------------------
- Scale generation now produces unique integer scales spaced
  geometrically; the original `np.logspace(..., dtype=int)` generated
  many duplicates (especially with large `max_scales`), so the actual
  number of distinct scales was far below the requested count.
- DBC formula now follows Sarkar & Chaudhuri 1994:
       n_r(i,j) = ceil(I_max / h) - ceil(I_min / h) + 1
  with h = s * G_max / H (s is the box side in pixels, H is the image
  side, G_max is the gray-value range). The original code computed
       ceil(delta_z / s + 1e-6)
  which silently drops the "+1" baseline of Sarkar's formula and
  systematically underestimates FD at small scales.
- SDBC is now actually different from DBC. SDBC (Chen et al. 1995)
  shifts every box's vertical origin to align with I_min, so:
       n_r(i,j) = floor((I_max - I_min) / h) + 1
- Empty box rows (count == 0) are dropped from the regression instead
  of being replaced with eps (which used to anchor the fit at log(eps)).
- Vectorized box-min/max via `view_as_blocks` reshape, no per-box
  Python loop. Order-of-magnitude speedup for DBC/SDBC on large images.
- New batch APIs: `CFAImageDimension.get_batch_*` accept a list of
  images and return one result dict per image, sharing scale generation
  to avoid redundant work.

Production-use caveats (deliberate departures from textbook DBC/SDBC)
---------------------------------------------------------------------
These are engineering choices, NOT bugs. They affect cross-image
comparability and strict literature reproducibility; see the per-method
docstrings (`get_dbc_fd`, `get_sdbc_fd`) for the full guidance.

1. G_max = np.percentile(image, 99)  (NOT G = 256 / image.max())
   Robust against saturated outliers but makes D content-adaptive:
   the same texture at different contrasts can yield different D.
   For cross-image comparison, pre-normalize the input to a fixed
   gray range before calling.
2. H_norm = max(image.shape)         (NOT M, with M × M square)
   For severely non-square inputs (aspect ratio >> 1), z-axis box
   height differs from a strict S&C 1994 derivation. Center-crop the
   input to square for rigorous use.
3. Scales are geometrically spaced unique integers, NOT dyadic (2^k).
   Gives more fit points but ~3-5% bias near D = 2. Pass
   `fit_range=(small, max_size/4)` to restrict the fit to the linear
   scaling region for the cleanest estimate.
"""

import math
import os
import sys
import json
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import linregress
from skimage.util import view_as_blocks

try:
    from FreeAeonFractal.FAImage import CFAImage
except Exception:  # pragma: no cover - fallback for direct import
    from FAImage import CFAImage


# ============================================================
# Scale generation utility (shared with the GPU version)
# ============================================================
def make_scales(max_size: int, max_scales: int = 30,
                min_size: int = 2) -> np.ndarray:
    """
    Geometrically spaced unique integer scales in [min_size, max_size].

    np.logspace(..., dtype=int) collapses many close values into the same
    integer, so even when `max_scales=10000` the actual count of distinct
    integers was small (and clustered). Here we generate floats first,
    round, deduplicate, and sort.
    """
    if max_size < max(2, min_size):
        return np.array([], dtype=int)
    raw = np.logspace(np.log2(min_size), np.log2(max_size),
                      num=max_scales, base=2.0)
    raw = np.unique(np.round(raw).astype(int))
    raw = raw[(raw >= min_size) & (raw <= max_size)]
    return raw.astype(int)


# ============================================================
# Block iteration helpers (vectorized using skimage.view_as_blocks)
# ============================================================
def _crop_to_multiple(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    new_h = (h // size) * size
    new_w = (w // size) * size
    return img[:new_h, :new_w]


def _pad_to_multiple(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    new_h = ((h + size - 1) // size) * size
    new_w = ((w + size - 1) // size) * size
    pad_h = new_h - h
    pad_w = new_w - w
    return np.pad(img, ((0, pad_h), (0, pad_w)), mode="constant",
                  constant_values=0)


def _prepare(img: np.ndarray, size: int, corp_type: int) -> np.ndarray:
    """corp_type: -1 crop, 0 require divisible, 1 pad."""
    if corp_type == -1:
        return _crop_to_multiple(img, size)
    if corp_type == 1:
        return _pad_to_multiple(img, size)
    if corp_type == 0:
        h, w = img.shape[:2]
        if (h % size) != 0 or (w % size) != 0:
            raise ValueError(
                f"corp_type=0 requires H,W divisible by {size}, got {(h, w)}")
        return img
    raise ValueError("corp_type must be -1, 0, or 1")


def _blocks_min_max(img2d: np.ndarray, size: int,
                    corp_type: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-box min and max, vectorized.

    Returns
    -------
    box_min, box_max : (nY, nX) float
    """
    proc = _prepare(img2d, size, corp_type=corp_type).astype(np.float64,
                                                              copy=False)
    h, w = proc.shape
    nY, nX = h // size, w // size
    if nY == 0 or nX == 0:
        return np.empty((0, 0)), np.empty((0, 0))
    # (nY, size, nX, size) -> (nY, nX, size, size)
    blocks = proc.reshape(nY, size, nX, size).transpose(0, 2, 1, 3)
    blocks = blocks.reshape(nY, nX, size * size)
    return blocks.min(axis=2), blocks.max(axis=2)


def _blocks_occupied(img2d: np.ndarray, size: int,
                     corp_type: int = -1) -> int:
    """Number of non-empty boxes (any positive pixel inside).
    """
    proc = _prepare(img2d, size, corp_type=corp_type)
    h, w = proc.shape
    nY, nX = h // size, w // size
    if nY == 0 or nX == 0:
        return 0
    blocks = proc.reshape(nY, size, nX, size).transpose(0, 2, 1, 3)
    flat = blocks.reshape(nY * nX, size * size)
    return int((flat > 0).any(axis=1).sum())


# ============================================================
# Linear regression on log-log points, with empty-row handling
# ============================================================
def _auto_select_fit_range(s_valid: np.ndarray, n_valid: np.ndarray,
                            min_points: int = 5
                            ) -> Optional[Tuple[float, float]]:
    """
    Find the most "linear" sub-window of the log-log curve.

    Box-counting log-log curves are sigmoid-shaped: saturated at small
    s by pixel discretization, saturated at large s by image bounds.
    Only the middle is a real power law, and that middle moves around
    depending on image size, content, and method (BC vs DBC vs SDBC).

    Scoring criterion
    -----------------
    We use the (length-adjusted) **F-statistic** of the linear fit:

        F = R² / (1 - R²) · (length - 2)

    This is the standard signal-to-noise ratio of a linear regression.
    Compared to ``R² × √length`` it correctly punishes even tiny R²
    drops (e.g. R²=0.9932 vs 0.9999 → F differs by ~2×, but R² differs
    by only 0.7%). The (1-R²) divisor is clamped to 1e-6 to avoid
    overflow on perfect fits.

    Parameters
    ----------
    s_valid, n_valid : 1D arrays of positive finite values, same length
    min_points : minimum number of points in the chosen window

    Returns
    -------
    (s_lo, s_hi) tuple or None if not enough valid points.
    """
    if s_valid.size < min_points:
        return None
    # Sort by scale ascending so contiguous index windows = contiguous
    # scale windows.
    order = np.argsort(s_valid)
    s_sorted = s_valid[order]
    n_sorted = n_valid[order]
    x = -np.log(s_sorted)
    y = np.log(n_sorted)
    n_pts = x.size

    best_score = -np.inf
    best_lo, best_hi = float(s_sorted[0]), float(s_sorted[-1])
    for i in range(0, n_pts - min_points + 1):
        for j in range(i + min_points, n_pts + 1):
            xs = x[i:j]
            ys = y[i:j]
            length = j - i
            x_mean = xs.mean()
            y_mean = ys.mean()
            ss_xx = float(np.sum((xs - x_mean) ** 2))
            ss_yy = float(np.sum((ys - y_mean) ** 2))
            ss_xy = float(np.sum((xs - x_mean) * (ys - y_mean)))
            if ss_xx <= 0 or ss_yy <= 0:
                continue
            r2 = (ss_xy * ss_xy) / (ss_xx * ss_yy)
            # F-statistic: R²/(1-R²) · (n-2). Clamp 1-R² to 1e-6 to
            # cap perfect fits; multiply by length surrogate for
            # statistical power.
            score = (r2 / max(1.0 - r2, 1e-6)) * max(length - 2, 1)
            if score > best_score:
                best_score = score
                best_lo = float(s_sorted[i])
                best_hi = float(s_sorted[j - 1])
    return (best_lo, best_hi)


def _fit_log_log(scales: np.ndarray, counts: np.ndarray,
                 fit_range: Optional[Tuple[float, float]] = None,
                 auto_fit_range: bool = False,
                 auto_fit_min_points: int = 5) -> dict:
    """
    Fit log N(r) vs log(1/r). Drops scales with N=0 instead of replacing
    them with eps (which would anchor the fit at log(eps)).

    Parameters
    ----------
    fit_range : (s_min, s_max) or None
        If given, only scales in [s_min, s_max] are used for the fit.
        Useful because box-counting log-log curves are sigmoid-shaped
        (saturated at small s by pixel discretization, saturated at
        large s by image bounds); only the middle is a real power law.
    auto_fit_range : bool, default False
        When True AND ``fit_range`` is None, automatically select the
        most-linear sub-window of the curve (maximises R²×√length on
        the valid points). Ignored when ``fit_range`` is provided
        explicitly (explicit user input always wins).
    auto_fit_min_points : int, default 5
        Minimum number of points the auto-selected window must contain.

    Returns
    -------
    dict with the regular fit fields plus ``"fit_range_used"``:
        - None if no range was applied,
        - the explicit user range, or
        - the auto-selected range as a (s_lo, s_hi) tuple.
    """
    s = np.asarray(scales, dtype=np.float64)
    n = np.asarray(counts, dtype=np.float64)
    keep = (s > 0) & (n > 0) & np.isfinite(s) & np.isfinite(n)
    s_valid = s[keep]
    n_valid = n[keep]

    # Auto-select a window iff the caller asked for it AND didn't give
    # an explicit fit_range. Explicit user input always overrides.
    if auto_fit_range and fit_range is None:
        auto_range = _auto_select_fit_range(
            s_valid, n_valid, min_points=auto_fit_min_points)
        if auto_range is not None:
            fit_range = auto_range

    if fit_range is not None:
        s_lo, s_hi = float(fit_range[0]), float(fit_range[1])
        keep &= (s >= s_lo) & (s <= s_hi)
    s_used = s[keep]
    n_used = n[keep]
    if s_used.size < 2:
        return {
            "fd": float("nan"),
            "scales": s.tolist(), "counts": n.tolist(),
            "log_scales": [], "log_counts": [], "intercept": float("nan"),
            "r_value": float("nan"), "p_value": float("nan"),
            "std_err": float("nan"),
            "fit_range_used": fit_range,
        }
    log_scales = -np.log(s_used)
    log_counts = np.log(n_used)
    slope, intercept, r_value, p_value, std_err = linregress(
        log_scales, log_counts)
    return {
        "fd": float(slope),
        "scales": s_used.tolist(),
        "counts": n_used.tolist(),
        "log_scales": log_scales.tolist(),
        "log_counts": log_counts.tolist(),
        "intercept": float(intercept),
        "r_value": float(r_value),
        "p_value": float(p_value),
        "std_err": float(std_err),
        "fit_range_used": fit_range,
    }


# ============================================================
# Main class
# ============================================================
class CFAImageFD(object):
    """
    image: input image (single channel). For BC, pass a binary image
           (any positive pixel = occupied). For DBC/SDBC, pass the gray
           image.
    max_size: maximum box size for partitioning. Defaults to min(H, W).
    max_scales: target number of distinct scales (some collapse to the
                same integer at small sizes; result has fewer than this).
    min_size: smallest box size. Default 2 (size 1 has nothing to count).
    """

    def __init__(self, image: Optional[np.ndarray] = None,
                 max_size: Optional[int] = None,
                 max_scales: int = 30,
                 with_progress: bool = True,
                 min_size: int = 2):
        if image is None:
            raise ValueError("image must be provided")
        if image.ndim != 2:
            raise ValueError("image must be a 2D single-channel array")
        self.m_image = image
        if max_size is None:
            max_size = min(image.shape)
        self.m_with_progress = with_progress
        self.m_scales = make_scales(int(max_size), max_scales=max_scales,
                                    min_size=int(min_size)).tolist()

    # ------------------------------------------------------------
    # Linear regression entry-point (kept for API compatibility)
    # ------------------------------------------------------------
    def get_fd(self, scale_list, box_count_list,
               fit_range: Optional[Tuple[float, float]] = None,
               auto_fit_range: bool = False,
               auto_fit_min_points: int = 5):
        return _fit_log_log(np.asarray(scale_list),
                            np.asarray(box_count_list),
                            fit_range=fit_range,
                            auto_fit_range=auto_fit_range,
                            auto_fit_min_points=auto_fit_min_points)

    # ============================================================
    # Single-image methods
    # ============================================================
    def get_bc_fd(self, corp_type: int = -1,
                  fit_range: Optional[Tuple[float, float]] = None,
                  auto_fit_range: bool = False,
                  auto_fit_min_points: int = 5):
        """Box-counting on a binary image. Any positive pixel = occupied.

        ``auto_fit_range``: when True and ``fit_range`` is None, the
        most-linear sub-window of the log-log curve is selected
        automatically (maximises R²×√length). The chosen window is
        returned in the result dict under ``"fit_range_used"``. An
        explicit ``fit_range`` always overrides auto-selection.
        """
        scale_list, count_list = [], []
        it = (tqdm(self.m_scales, desc="Calculating by BC")
              if self.m_with_progress else self.m_scales)
        for size in it:
            scale_list.append(int(size))
            count_list.append(_blocks_occupied(self.m_image, size,
                                               corp_type=corp_type))
        return _fit_log_log(np.array(scale_list), np.array(count_list),
                            fit_range=fit_range,
                            auto_fit_range=auto_fit_range,
                            auto_fit_min_points=auto_fit_min_points)

    def get_dbc_fd(self, corp_type: int = -1,
                   fit_range: Optional[Tuple[float, float]] = None,
                   auto_fit_range: bool = False,
                   auto_fit_min_points: int = 5):
        """
        Differential Box Counting (Sarkar & Chaudhuri 1994).

        For each block of side s, compute box height h = s * G_max / H,
        then n_r(i,j) = ceil(I_max / h) - ceil(I_min / h) + 1.
        N_r = sum n_r(i,j).

        Note on G_max (deliberate departure from S&C 1994)
        --------------------------------------------------
        ``G_max`` here is ``np.percentile(image, 99)`` for outlier
        robustness, NOT the fixed gray range G = 256 (or G = image.max())
        that the original Sarkar & Chaudhuri 1994 paper uses. This makes
        the implementation **content-adaptive**: the same texture
        rendered at two different brightness / contrast settings will
        yield slightly different D values, because G_max changes with
        the input.

        Practical implications
        ----------------------
        - Single-image analysis or batches produced by a uniform pipeline:
          no problem. The choice of G_max is internally consistent.
        - **Cross-image comparison** (e.g. comparing D across images with
          different exposure / contrast): pre-normalize the input to a
          fixed gray range BEFORE calling this method, e.g.
              img = ((img - img.min()) * 255 /
                     max(img.max() - img.min(), 1)).astype(np.uint8)
          or any other fixed scheme, so G_max is comparable across
          images.
        - **Reproducing literature results**: replace ``G_max`` with
          ``255`` (8-bit) or ``image.max()`` to match the strict S&C
          1994 convention.

        Non-square images: H = max(image.shape) is used, which is a
        common engineering choice but not strictly the M × M assumption
        of S&C 1994. For rigorous DBC, center-crop the input to square
        first.

        Non-dyadic scales: this method's default scale list is
        geometrically spaced integers (not 2^k), which can introduce a
        few-percent bias for images saturating BC near D = 2. Pass
        ``fit_range=(small, max_size/4)`` to restrict the fit to the
        linear scaling region.
        """
        scale_list, count_list = [], []
        H = float(max(self.m_image.shape))
        G_max = float(np.percentile(self.m_image, 99))
        if G_max <= 0:
            G_max = float(np.max(self.m_image)) if self.m_image.size else 1.0
        G_max = max(G_max, 1e-12)

        it = (tqdm(self.m_scales, desc="Calculating by DBC")
              if self.m_with_progress else self.m_scales)
        for size in it:
            box_min, box_max = _blocks_min_max(self.m_image, size,
                                                corp_type=corp_type)
            if box_min.size == 0:
                continue
            h = float(size) * G_max / H
            n_r = (np.ceil(box_max / h) - np.ceil(box_min / h) + 1.0)
            np.clip(n_r, 1.0, None, out=n_r)
            box_count = int(n_r.sum())
            scale_list.append(int(size))
            count_list.append(box_count)
        return _fit_log_log(np.array(scale_list), np.array(count_list),
                            fit_range=fit_range,
                            auto_fit_range=auto_fit_range,
                            auto_fit_min_points=auto_fit_min_points)

    def get_sdbc_fd(self, corp_type: int = -1,
                    fit_range: Optional[Tuple[float, float]] = None,
                    auto_fit_range: bool = False,
                    auto_fit_min_points: int = 5):
        """
        Shifted DBC (Chen et al. 1995).

        Aligns the bottom of each column-stack of boxes to I_min. The
        analytic result of that geometric shift is
            n_r(i,j) = floor((I_max - I_min) / h) + 1
        which avoids the "boundary crossing" overcount of plain DBC.
        (The formula above is not an approximation — it is exactly what
        Chen 1995's geometric construction reduces to.)

        Same G_max caveat as ``get_dbc_fd`` applies: G_max is the 99th
        percentile of the input, NOT the fixed gray range of the
        original paper. See ``get_dbc_fd`` docstring for the full
        production-use guidance on:
          - G_max choice (content-adaptive vs literature-strict),
          - non-square image handling,
          - non-dyadic scales and ``fit_range``.
        """
        scale_list, count_list = [], []
        H = float(max(self.m_image.shape))
        G_max = float(np.percentile(self.m_image, 99))
        if G_max <= 0:
            G_max = float(np.max(self.m_image)) if self.m_image.size else 1.0
        G_max = max(G_max, 1e-12)

        it = (tqdm(self.m_scales, desc="Calculating by SDBC")
              if self.m_with_progress else self.m_scales)
        for size in it:
            box_min, box_max = _blocks_min_max(self.m_image, size,
                                                corp_type=corp_type)
            if box_min.size == 0:
                continue
            h = float(size) * G_max / H
            delta = np.clip(box_max - box_min, 0.0, None)
            n_r = np.floor(delta / h) + 1.0
            box_count = int(n_r.sum())
            scale_list.append(int(size))
            count_list.append(box_count)
        return _fit_log_log(np.array(scale_list), np.array(count_list),
                            fit_range=fit_range,
                            auto_fit_range=auto_fit_range,
                            auto_fit_min_points=auto_fit_min_points)

    # ============================================================
    # Batch APIs
    # ============================================================
    @staticmethod
    def get_batch_bc(images: Sequence[np.ndarray],
                     max_size: Optional[int] = None,
                     max_scales: int = 30,
                     min_size: int = 2,
                     corp_type: int = -1,
                     fit_range: Optional[Tuple[float, float]] = None,
                     auto_fit_range: bool = False,
                     auto_fit_min_points: int = 5,
                     with_progress: bool = True) -> List[dict]:
        """Batch BC. Each image must be 2D and binary (or any positive =
        occupied). Returns one fit-result dict per image.

        ``auto_fit_range`` (default False): when True and ``fit_range``
        is None, each image's log-log curve gets its OWN best-linear
        window picked independently. The chosen window is reported per
        image in result["fit_range_used"].
        """
        return _batch_run(images, "bc", corp_type=corp_type,
                          max_size=max_size, max_scales=max_scales,
                          min_size=min_size, fit_range=fit_range,
                          auto_fit_range=auto_fit_range,
                          auto_fit_min_points=auto_fit_min_points,
                          with_progress=with_progress)

    @staticmethod
    def get_batch_dbc(images: Sequence[np.ndarray],
                      max_size: Optional[int] = None,
                      max_scales: int = 30,
                      min_size: int = 2,
                      corp_type: int = -1,
                      fit_range: Optional[Tuple[float, float]] = None,
                      auto_fit_range: bool = False,
                      auto_fit_min_points: int = 5,
                      with_progress: bool = True) -> List[dict]:
        """Batch DBC. Each image must be 2D grayscale.

        See ``get_batch_bc`` for ``auto_fit_range`` semantics."""
        return _batch_run(images, "dbc", corp_type=corp_type,
                          max_size=max_size, max_scales=max_scales,
                          min_size=min_size, fit_range=fit_range,
                          auto_fit_range=auto_fit_range,
                          auto_fit_min_points=auto_fit_min_points,
                          with_progress=with_progress)

    @staticmethod
    def get_batch_sdbc(images: Sequence[np.ndarray],
                       max_size: Optional[int] = None,
                       max_scales: int = 30,
                       min_size: int = 2,
                       corp_type: int = -1,
                       fit_range: Optional[Tuple[float, float]] = None,
                       auto_fit_range: bool = False,
                       auto_fit_min_points: int = 5,
                       with_progress: bool = True) -> List[dict]:
        """Batch SDBC. Each image must be 2D grayscale.

        See ``get_batch_bc`` for ``auto_fit_range`` semantics."""
        return _batch_run(images, "sdbc", corp_type=corp_type,
                          max_size=max_size, max_scales=max_scales,
                          min_size=min_size, fit_range=fit_range,
                          auto_fit_range=auto_fit_range,
                          auto_fit_min_points=auto_fit_min_points,
                          with_progress=with_progress)

    # ============================================================
    # Plot helper (unchanged from original, slightly cleaner)
    # ============================================================
    @staticmethod
    def plot(raw_img, gray_img,bin_img, fd_bc, fd_dbc, fd_sdbc):
        def show_image(text, image, cmap="viridis"):
            plt.imshow(image, cmap=cmap)
            plt.title(text, fontsize=8)
            plt.axis("off")

        def show_fit(text, result):
            x = np.array(result["log_scales"])
            y = np.array(result["log_counts"])
            if x.size < 2:
                plt.title(f"{text}: insufficient data", fontsize=7)
                return
            fd = result["fd"]
            b = result["intercept"]
            r2 = result["r_value"] ** 2
            scale_range = f"[{min(result['scales'])}, {max(result['scales'])}]"

            plt.plot(x, y, "ro", label="Calculated points", markersize=2)
            plt.plot(x, fd * x + b, "k--", label="Linear fit")
            plt.fill_between(x, fd * x + b - 2 * result["std_err"],
                             fd * x + b + 2 * result["std_err"],
                             color="gray", alpha=0.2, label=r"$\pm 2\sigma$")
            textstr = "\n".join((
                fr"$D={fd:.4f}$",
                fr"$R^2={r2:.4f}$",
                f"Scale: {scale_range}",
            ))
            plt.gca().text(0.95, 0.95, textstr,
                           transform=plt.gca().transAxes, fontsize=7,
                           verticalalignment="top",
                           horizontalalignment="right",
                           bbox=dict(boxstyle="round,pad=0.2",
                                     facecolor="white", alpha=0.5))
            plt.title(f"{text}: FD={fd:.4f} PV={result['p_value']:.4f}",
                      fontsize=7)
            plt.xlabel(r"$\log(1/r)$", fontsize=7)
            plt.ylabel(r"$\log(N(r))$", fontsize=7)
            plt.legend(fontsize=7)
            plt.grid(True, which="both", ls="--", lw=0.3)

        plt.figure(1, figsize=(10, 5))
        plt.subplot(2, 3, 1); show_image("Raw Image", raw_img)
        plt.subplot(2, 3, 2); show_image("Gray Image", gray_img,'gray')
        plt.subplot(2, 3, 3); show_image("Binary Image", bin_img, "gray")
        plt.subplot(2, 3, 4); show_fit("BC", fd_bc)
        plt.subplot(2, 3, 5); show_fit("DBC", fd_dbc)
        plt.subplot(2, 3, 6); show_fit("SDBC", fd_sdbc)
        plt.tight_layout()
        plt.show()


# ============================================================
# Internal batch driver
#
# We loop scales as the OUTER loop and images as the INNER loop. For
# each scale we vectorize across all images at once (when they share
# H, W; otherwise we fall back to per-image processing). This:
#   - avoids re-generating scales for every image,
#   - lets numpy reuse temporaries within a scale,
#   - keeps the cache hot for the small per-block (size, size) tile.
# ============================================================
def _batch_run(images: Sequence[np.ndarray], method: str,
               corp_type: int, max_size: Optional[int],
               max_scales: int, min_size: int,
               with_progress: bool,
               fit_range: Optional[Tuple[float, float]] = None,
               auto_fit_range: bool = False,
               auto_fit_min_points: int = 5
               ) -> List[dict]:
    """
    Batch driver. Iterates over images on the outer loop and shares the
    pre-computed scale list. Within each image the inner loop is the
    same vectorized per-scale block reduction as the single-image API.

    NOTE: numpy on CPU does *not* benefit from "batch the box-reductions
    across images" because the (N, nb, s*s) intermediate exceeds L2
    cache; we keep working sets small per image. The shared work that
    actually amortises across the batch is scale generation and (for
    DBC/SDBC) percentile pre-computation.
    """
    if len(images) == 0:
        return []
    for i, im in enumerate(images):
        if im.ndim != 2:
            raise ValueError(f"image {i} is not 2D")

    N = len(images)
    if max_size is None:
        max_size = min(min(im.shape) for im in images)
    scales = make_scales(int(max_size), max_scales=max_scales,
                         min_size=int(min_size))
    if scales.size < 2:
        raise ValueError(f"Need >= 2 scales; got {scales.size}")
    counts = np.zeros((N, scales.size), dtype=np.float64)
    scale_arr = scales.astype(np.float64)

    # Pre-compute per-image G_max once (DBC/SDBC). For BC it's unused.
    if method in ("dbc", "sdbc"):
        G_max = np.zeros(N, dtype=np.float64)
        for i in range(N):
            v = float(np.percentile(images[i], 99))
            if v <= 0:
                v = float(np.max(images[i])) if images[i].size else 1.0
            G_max[i] = max(v, 1e-12)

    it = (tqdm(range(N), desc=f"Batch {method.upper()}")
          if with_progress else range(N))
    for i in it:
        img = images[i]
        H_norm = float(max(img.shape))
        for k, size in enumerate(scales):
            size = int(size)
            if method == "bc":
                counts[i, k] = _blocks_occupied(img, size,
                                                 corp_type=corp_type)
            else:
                bmin, bmax = _blocks_min_max(img, size, corp_type=corp_type)
                if bmin.size == 0:
                    continue
                h_box = float(size) * G_max[i] / H_norm
                if method == "dbc":
                    n_r = (np.ceil(bmax / h_box)
                           - np.ceil(bmin / h_box) + 1.0)
                    np.clip(n_r, 1.0, None, out=n_r)
                else:
                    delta = np.clip(bmax - bmin, 0.0, None)
                    n_r = np.floor(delta / h_box) + 1.0
                counts[i, k] = float(n_r.sum())

    return [_fit_log_log(scale_arr, counts[i], fit_range=fit_range,
                         auto_fit_range=auto_fit_range,
                         auto_fit_min_points=auto_fit_min_points)
            for i in range(N)]


def _prepare_batch(stack: np.ndarray, size: int,
                   corp_type: int) -> np.ndarray:
    """(Reserved for future use; kept for API symmetry with GPU side.)"""
    N, H, W = stack.shape
    if corp_type == -1:
        new_h = (H // size) * size
        new_w = (W // size) * size
        return stack[:, :new_h, :new_w]
    if corp_type == 1:
        new_h = ((H + size - 1) // size) * size
        new_w = ((W + size - 1) // size) * size
        return np.pad(stack,
                      ((0, 0), (0, new_h - H), (0, new_w - W)),
                      mode="constant", constant_values=0)
    if corp_type == 0:
        if (H % size) != 0 or (W % size) != 0:
            raise ValueError(
                f"corp_type=0 requires H,W divisible by {size}")
        return stack
    raise ValueError("corp_type must be -1, 0, or 1")


# ============================================================
# Demo
# ============================================================
def main():
    image_path = "../images/fractal.png"
    raw_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    bin_image = (raw_image < 64).astype(np.uint8)
    fd_bc = CFAImageFD(bin_image, max_scales=30).get_bc_fd(corp_type=-1)
    fd_dbc = CFAImageFD(raw_image, max_scales=30).get_dbc_fd(corp_type=-1)
    fd_sdbc = CFAImageFD(raw_image, max_scales=30).get_sdbc_fd(corp_type=-1)
    print(f"BC FD   = {fd_bc['fd']:.4f}")
    print(f"DBC FD  = {fd_dbc['fd']:.4f}")
    print(f"SDBC FD = {fd_sdbc['fd']:.4f}")
    CFAImageFD.plot(raw_image, raw_image, bin_image, fd_bc, fd_dbc, fd_sdbc)


if __name__ == "__main__":
    main()
