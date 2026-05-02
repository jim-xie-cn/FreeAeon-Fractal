"""
GPU-accelerated lacunarity analysis (single + batch).

See FAImageLacunarity.py for the algorithmic notes; this file uses the
identical math (Allain & Cloitre's gliding-box, Lambda = <M^2>/<M>^2),
implemented in PyTorch.

What changed vs the original GPU version
----------------------------------------
- Per-scale work is fully vectorised across the whole batch: the
  integral image is computed once for the (B, H+1, W+1) tensor, and
  every reduction (sum, sum-of-squares, count) is a single op over
  the whole batch instead of a Python loop with `.item()` calls.
- The original `if not include_zero: torch.stack([m[m>0] for m in masses])`
  used to crash when different images contained different counts of
  zero-mass boxes; the new version handles unequal counts with masked
  reductions.
- Mass extraction uses tensor slicing (`S[:, s:, s:] - ...`) instead
  of `meshgrid`, which materialised an (H-s+1)*(W-s+1) integer-index
  tensor every call.
- Added explicit chunking via `img_chunk` so a giant batch can be
  streamed through GPU memory.
- Top-of-file `class CFAImage` shim has been removed; the Otsu helper
  comes from the project's CFAImage when available, else falls back
  to OpenCV directly in the demo.
- Same fit transforms ("log" and "log_minus_1") and `fit_range` as the
  CPU version.
"""

import math
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import linregress


# ============================================================
# Scale generation (identical semantics to the CPU file)
# ============================================================
def make_scales(max_size: int,
                scales_mode: str = "powers",
                max_scales: int = 100,
                min_size: int = 2) -> List[int]:
    max_size = int(max_size)
    min_size = int(max(1, min_size))
    if max_size < min_size:
        return []
    if scales_mode == "powers":
        max_pow = int(np.floor(np.log2(max_size)))
        return [int(2 ** k) for k in range(0, max_pow + 1)
                if 2 ** k >= min_size and 2 ** k <= max_size]
    if scales_mode == "logspace":
        raw = np.logspace(np.log2(min_size), np.log2(max_size),
                          num=int(max_scales), base=2.0)
        ints = np.unique(np.round(raw).astype(int))
        ints = ints[(ints >= min_size) & (ints <= max_size)]
        return ints.astype(int).tolist()
    raise ValueError(f"unknown scales_mode={scales_mode!r}")


# ============================================================
# Curve fitting (mirror of CPU)
# ============================================================
def _fit_lacunarity_curve(scales: np.ndarray, lacs: np.ndarray,
                          transform: str = "log",
                          fit_range: Optional[Tuple[float, float]] = None
                          ) -> dict:
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
        "log_lambda_minus_1": y_fit.tolist(),
        "transform": transform,
    }


# ============================================================
# Main class
# ============================================================
class CFAImageLACGPU:
    """
    GPU lacunarity calculator. Single-image mode mirrors the CPU API;
    batch mode is exposed via the static `get_batch_lacunarity` method.
    """

    def __init__(self,
                 image: Optional[np.ndarray] = None,
                 max_size: Optional[int] = None,
                 max_scales: int = 100,
                 with_progress: bool = True,
                 scales_mode: str = "powers",
                 partition_mode: str = "gliding",
                 min_size: int = 2,
                 device: Optional[Union[str, torch.device]] = None,
                 dtype: torch.dtype = torch.float64):
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
                                   max_scales=max_scales,
                                   min_size=min_size)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype

    # ------------------------------------------------------------
    # Single-image lacunarity
    # ------------------------------------------------------------
    @torch.no_grad()
    def get_lacunarity(self, corp_type: int = -1,
                       use_binary_mass: bool = False,
                       include_zero: bool = True) -> dict:
        # Single image is run through the batch path with B=1, which
        # uses the same vectorised reductions.
        results = _batch_lacunarity_gpu(
            [self.image], scales=self.scales,
            partition_mode=self.partition_mode,
            use_binary_mass=use_binary_mass,
            include_zero=include_zero,
            device=self.device, dtype=self.dtype,
            img_chunk=1,
            with_progress=self.with_progress)
        return results[0]

    # ------------------------------------------------------------
    # Curve fit (instance API for back-compat)
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
    # Batch APIs
    # ============================================================
    @staticmethod
    @torch.no_grad()
    def get_batch_lacunarity(images: Sequence[np.ndarray],
                             max_size: Optional[int] = None,
                             max_scales: int = 100,
                             scales_mode: str = "powers",
                             partition_mode: str = "gliding",
                             min_size: int = 2,
                             use_binary_mass: bool = False,
                             include_zero: bool = True,
                             device: Optional[Union[str, torch.device]] = None,
                             dtype: torch.dtype = torch.float64,
                             img_chunk: Optional[int] = None,
                             with_progress: bool = True) -> List[dict]:
        if len(images) == 0:
            return []
        for i, im in enumerate(images):
            if im.ndim != 2:
                raise ValueError(f"image {i} is not 2D")
        if not all(im.shape == images[0].shape for im in images):
            raise ValueError(
                "All images must share the same shape on the GPU batch "
                "path.")
        if max_size is None:
            max_size = min(min(im.shape) for im in images)
        scales = make_scales(max_size, scales_mode=scales_mode,
                             max_scales=max_scales, min_size=min_size)
        if not scales:
            raise ValueError("No valid scales generated")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        if img_chunk is None or img_chunk <= 0:
            img_chunk = len(images)
        return _batch_lacunarity_gpu(
            images, scales=scales,
            partition_mode=partition_mode,
            use_binary_mass=use_binary_mass,
            include_zero=include_zero,
            device=device, dtype=dtype,
            img_chunk=int(img_chunk),
            with_progress=with_progress)

    @staticmethod
    def fit_batch_lacunarity(lac_results: Sequence[dict],
                             transform: str = "log",
                             fit_range: Optional[Tuple[float, float]] = None
                             ) -> List[dict]:
        return [_fit_lacunarity_curve(np.asarray(r["scales"]),
                                       np.asarray(r["lacunarity"]),
                                       transform=transform,
                                       fit_range=fit_range)
                for r in lac_results]

    # ------------------------------------------------------------
    # Plotting (mirror of CPU)
    # ------------------------------------------------------------
    @staticmethod
    def plot(lac_result, fit_result=None, ax=None, show=True,
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
        ax_main.set_xscale("log"); ax_main.set_yscale("log")
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
                ax_fit.plot(xline, slope * xline + intercept, "r-", lw=2,
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
# Internal: fully vectorised batch driver
# ============================================================
@torch.no_grad()
def _batch_lacunarity_gpu(images: Sequence[np.ndarray],
                          scales: List[int],
                          partition_mode: str,
                          use_binary_mass: bool,
                          include_zero: bool,
                          device: torch.device,
                          dtype: torch.dtype,
                          img_chunk: int,
                          with_progress: bool) -> List[dict]:
    N = len(images)
    H, W = images[0].shape

    out_per_image: List[dict] = [{
        "scales": [],
        "lacunarity": [],
        "mass_stats": [],
    } for _ in range(N)]

    chunk_iter = range(0, N, img_chunk)
    chunk_iter = (tqdm(chunk_iter, total=(N + img_chunk - 1) // img_chunk,
                       desc=f"Batch lacunarity ({partition_mode})")
                  if with_progress else chunk_iter)

    for lo in chunk_iter:
        hi = min(N, lo + img_chunk)
        b = hi - lo

        # Stage chunk on device
        # (We accept Python list -> torch directly; cheap for small b.)
        chunk_np = np.stack([im.astype(np.float64, copy=False)
                             for im in images[lo:hi]], axis=0)
        chunk = torch.from_numpy(chunk_np).to(device=device, dtype=dtype)
        if use_binary_mass:
            chunk = (chunk > 0).to(dtype)

        if partition_mode == "gliding":
            # Integral image, computed ONCE for the whole chunk.
            # S has shape (b, H+1, W+1) with leading zero row/col so
            # that mass(y, x; s) = S[y+s, x+s] - S[y, x+s]
            #                       - S[y+s, x] + S[y, x]
            S = torch.zeros((b, H + 1, W + 1), device=device, dtype=dtype)
            torch.cumsum(torch.cumsum(chunk, dim=1), dim=2, out=S[:, 1:, 1:])
            for size in scales:
                _glide_one_scale_into(S, size, b, H, W,
                                       include_zero=include_zero,
                                       out_per_image=out_per_image,
                                       offset=lo)
            del S
        else:
            # non-overlapping: at each scale, reshape into block tiles.
            for size in scales:
                _nonov_one_scale_into(chunk, size, b, H, W,
                                       include_zero=include_zero,
                                       out_per_image=out_per_image,
                                       offset=lo)

        del chunk
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return out_per_image


def _glide_one_scale_into(S: torch.Tensor, size: int,
                          b: int, H: int, W: int,
                          include_zero: bool,
                          out_per_image: List[dict],
                          offset: int) -> None:
    if size > H or size > W:
        for j in range(b):
            out_per_image[offset + j]["scales"].append(int(size))
            out_per_image[offset + j]["lacunarity"].append(float("nan"))
            out_per_image[offset + j]["mass_stats"].append({
                "scale": int(size),
                "num_boxes": 0,
                "mean_mass": float("nan"),
                "var_mass": float("nan"),
                "lambda": float("nan"),
            })
        return

    # M: (b, H-s+1, W-s+1), all gliding boxes for the chunk.
    M = (S[:, size:, size:] - S[:, :-size, size:]
         - S[:, size:, :-size] + S[:, :-size, :-size])

    if include_zero:
        n_per = M.shape[1] * M.shape[2]
        sum_M = M.sum(dim=(1, 2))
        sum_M2 = (M * M).sum(dim=(1, 2))
        n_arr = torch.full((b,), n_per, device=M.device, dtype=torch.int64)
    else:
        mask = (M > 0)
        n_arr = mask.sum(dim=(1, 2))                                   # (b,)
        Mm = torch.where(mask, M, torch.zeros_like(M))
        sum_M = Mm.sum(dim=(1, 2))
        sum_M2 = (Mm * Mm).sum(dim=(1, 2))

    _push_chunk_results(out_per_image, n_arr, sum_M, sum_M2,
                        size=int(size), b=b, offset=offset)


def _nonov_one_scale_into(chunk: torch.Tensor, size: int,
                          b: int, H: int, W: int,
                          include_zero: bool,
                          out_per_image: List[dict],
                          offset: int) -> None:
    nY, nX = H // size, W // size
    if nY == 0 or nX == 0:
        for j in range(b):
            out_per_image[offset + j]["scales"].append(int(size))
            out_per_image[offset + j]["lacunarity"].append(float("nan"))
            out_per_image[offset + j]["mass_stats"].append({
                "scale": int(size),
                "num_boxes": 0,
                "mean_mass": float("nan"),
                "var_mass": float("nan"),
                "lambda": float("nan"),
            })
        return

    arr = chunk[:, :nY * size, :nX * size]
    blocks = arr.reshape(b, nY, size, nX, size).permute(0, 1, 3, 2, 4)
    M = blocks.reshape(b, nY * nX, size * size).sum(dim=2)             # (b,nb)

    if include_zero:
        n_per = M.shape[1]
        sum_M = M.sum(dim=1)
        sum_M2 = (M * M).sum(dim=1)
        n_arr = torch.full((b,), n_per, device=M.device, dtype=torch.int64)
    else:
        mask = (M > 0)
        n_arr = mask.sum(dim=1)
        Mm = torch.where(mask, M, torch.zeros_like(M))
        sum_M = Mm.sum(dim=1)
        sum_M2 = (Mm * Mm).sum(dim=1)

    _push_chunk_results(out_per_image, n_arr, sum_M, sum_M2,
                        size=int(size), b=b, offset=offset)


def _push_chunk_results(out_per_image: List[dict],
                        n_arr: torch.Tensor,
                        sum_M: torch.Tensor,
                        sum_M2: torch.Tensor,
                        size: int, b: int, offset: int) -> None:
    """
    Convert per-image (n_boxes, sum_M, sum_M^2) tensors to Lambda(r) and
    append per-image rows to the output list. One CPU sync per chunk.
    """
    n_safe = n_arr.clamp_min(1).to(sum_M.dtype)
    mean_M = sum_M / n_safe
    mean_M2 = sum_M2 / n_safe
    var_M = mean_M2 - mean_M * mean_M
    valid = (n_arr > 0) & (mean_M > 0) & torch.isfinite(mean_M2)
    lam = torch.where(valid,
                      mean_M2 / (mean_M * mean_M),
                      torch.full_like(mean_M, float("nan")))

    # One device->host transfer per scale per chunk
    n_np   = n_arr.detach().cpu().numpy()
    mean_np = torch.where(valid, mean_M, torch.full_like(mean_M, float("nan"))
                          ).detach().cpu().numpy()
    var_np = torch.where(valid, var_M, torch.full_like(var_M, float("nan"))
                          ).detach().cpu().numpy()
    lam_np = lam.detach().cpu().numpy()

    for j in range(b):
        out_per_image[offset + j]["scales"].append(size)
        out_per_image[offset + j]["lacunarity"].append(float(lam_np[j]))
        out_per_image[offset + j]["mass_stats"].append({
            "scale": size,
            "num_boxes": int(n_np[j]),
            "mean_mass": float(mean_np[j]),
            "var_mass": float(var_np[j]),
            "lambda": float(lam_np[j]),
        })


# ============================================================
# Demo
# ============================================================
def main():
    import cv2, os
    img_path = "../images/face.png"
    if not os.path.exists(img_path):
        rng = np.random.default_rng(0)
        gray_image = rng.integers(0, 256, size=(256, 256)).astype(np.uint8)
    else:
        gray_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, bin_image = cv2.threshold(gray_image, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Single image
    calc = CFAImageLACGPU(bin_image, scales_mode="powers",
                                  partition_mode="gliding",
                                  with_progress=False)
    res = calc.get_lacunarity(use_binary_mass=True)
    fit = calc.fit_lacunarity(res)
    print("Single Lambda(r):", res["lacunarity"])
    print("Single fit slope:", fit["slope"])

    # Batch
    imgs = [bin_image, bin_image, bin_image]
    batch = CFAImageLACGPU.get_batch_lacunarity(
        imgs, scales_mode="powers", partition_mode="gliding",
        use_binary_mass=True, with_progress=False)
    fits = CFAImageLACGPU.fit_batch_lacunarity(batch)
    print("Batch slopes:", [f["slope"] for f in fits])


if __name__ == "__main__":
    main()
