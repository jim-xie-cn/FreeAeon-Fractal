"""
GPU-accelerated fractal dimension calculations (BC / DBC / SDBC).

Mirrors FAImageDimension on CPU; see that file for the algorithmic
notes and references. Differences from the original GPU version:

- Scale generation uses make_scales() (no integer-collapse bug).
- DBC follows Sarkar & Chaudhuri 1994:
      n_r(i,j) = ceil(I_max / h) - ceil(I_min / h) + 1
- SDBC is now actually different from DBC (Chen et al. 1995):
      n_r(i,j) = floor((I_max - I_min) / h) + 1
- Empty box rows are dropped from the regression instead of being
  replaced with eps (which used to anchor the fit at log(eps)).
- Vectorized batch path: cropping is done once on the (N, H, W) tensor
  (no per-image Python loop), .item() calls inside the inner loop have
  been removed (no GPU<->CPU sync per image), and torch.quantile is
  used directly (no roundtrip through numpy / np.quantile).
- Image chunking via `img_chunk` to control VRAM usage.
"""

import math
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# Scale generation (same as CPU side; duplicated for self-containment)
# ============================================================
def make_scales(max_size: int, max_scales: int = 30,
                min_size: int = 2) -> np.ndarray:
    if max_size < max(2, min_size):
        return np.array([], dtype=int)
    raw = np.logspace(np.log2(min_size), np.log2(max_size),
                      num=max_scales, base=2.0)
    raw = np.unique(np.round(raw).astype(int))
    raw = raw[(raw >= min_size) & (raw <= max_size)]
    return raw.astype(int)


# ============================================================
# Linear regression on log-log points (no SciPy dependency)
# ============================================================
def _fit_log_log(scales_np: np.ndarray, counts_np: np.ndarray,
                 fit_range: Optional[Tuple[float, float]] = None) -> dict:
    s = np.asarray(scales_np, dtype=np.float64)
    n = np.asarray(counts_np, dtype=np.float64)
    keep = (s > 0) & (n > 0) & np.isfinite(s) & np.isfinite(n)
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
        }
    x = -np.log(s_used)
    y = np.log(n_used)
    n_pts = x.size
    x_mean = x.mean()
    y_mean = y.mean()
    ss_xx = float(np.sum((x - x_mean) ** 2))
    ss_xy = float(np.sum((x - x_mean) * (y - y_mean)))
    ss_yy = float(np.sum((y - y_mean) ** 2))
    if ss_xx <= 0:
        return {
            "fd": float("nan"),
            "scales": s_used.tolist(), "counts": n_used.tolist(),
            "log_scales": x.tolist(), "log_counts": y.tolist(),
            "intercept": float("nan"), "r_value": float("nan"),
            "p_value": float("nan"), "std_err": float("nan"),
        }
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    r_den = math.sqrt(ss_xx * ss_yy) if ss_yy > 0 else 0.0
    r_value = (ss_xy / r_den) if r_den > 0 else float("nan")
    y_hat = slope * x + intercept
    ssr = float(np.sum((y - y_hat) ** 2))
    if n_pts > 2:
        s_err = math.sqrt(ssr / (n_pts - 2))
        std_err = s_err / math.sqrt(ss_xx)
    else:
        std_err = float("nan")
    return {
        "fd": float(slope),
        "scales": s_used.tolist(),
        "counts": n_used.tolist(),
        "log_scales": x.tolist(),
        "log_counts": y.tolist(),
        "intercept": float(intercept),
        "r_value": float(r_value),
        "p_value": None,        # not computed; use scipy if needed
        "std_err": float(std_err),
    }


# ============================================================
# Tensor crop / pad / blockify helpers
# ============================================================
def _crop_pad_batch(tensor: torch.Tensor, size: int,
                    corp_type: int) -> torch.Tensor:
    """
    tensor: (..., H, W). Crop or pad the trailing two axes to the
    nearest multiple of `size`.
    """
    H, W = tensor.shape[-2], tensor.shape[-1]
    if corp_type == -1:
        new_H = (H // size) * size
        new_W = (W // size) * size
        return tensor[..., :new_H, :new_W]
    if corp_type == 1:
        new_H = ((H + size - 1) // size) * size
        new_W = ((W + size - 1) // size) * size
        pad_h = new_H - H
        pad_w = new_W - W
        # pad uses (..., last_dim_pad_left, right, second_last_left, right, ...)
        return F.pad(tensor, (0, pad_w, 0, pad_h),
                     mode="constant", value=0.0)
    if corp_type == 0:
        if (H % size) != 0 or (W % size) != 0:
            raise ValueError(
                f"corp_type=0 requires H,W divisible by {size}, "
                f"got {(H, W)}")
        return tensor
    raise ValueError("corp_type must be -1, 0, or 1")


def _blocks_NHW(tensor: torch.Tensor, size: int) -> torch.Tensor:
    """
    tensor: (N, H', W') with H', W' divisible by size.
    Returns blocks of shape (N, nb, size*size).
    """
    N, H, W = tensor.shape
    nY, nX = H // size, W // size
    return (tensor.reshape(N, nY, size, nX, size)
                  .permute(0, 1, 3, 2, 4)
                  .reshape(N, nY * nX, size * size))


# ============================================================
# Main class
# ============================================================
class CFAImageFDGPU:
    """
    GPU fractal dimension calculator. Single-image mode mirrors the CPU
    API; batch mode is exposed via the static `get_batch_*` methods.
    """

    def __init__(self,
                 image: Optional[np.ndarray] = None,
                 max_size: Optional[int] = None,
                 max_scales: int = 30,
                 with_progress: bool = True,
                 device: Optional[Union[str, torch.device]] = None,
                 dtype: torch.dtype = torch.float32,
                 min_size: int = 2):
        if image is not None:
            if image.ndim != 2:
                raise ValueError("image must be a 2D single-channel array")
            if max_size is None:
                max_size = min(image.shape)
        else:
            if max_size is None:
                raise ValueError(
                    "max_size required when constructing without image")

        self.m_with_progress = with_progress
        self.m_scales: List[int] = make_scales(
            int(max_size), max_scales=max_scales,
            min_size=int(min_size)).tolist()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype

        if image is not None:
            img_t = torch.from_numpy(np.asarray(image)).to(
                self.device, dtype=self.dtype)
            self.img = img_t                            # (H, W)
        else:
            self.img = None

    # ------------------------------------------------------------
    # API-compat wrapper
    # ------------------------------------------------------------
    def get_fd(self, scale_list, box_count_list,
               fit_range: Optional[Tuple[float, float]] = None):
        return _fit_log_log(np.asarray(scale_list),
                            np.asarray(box_count_list),
                            fit_range=fit_range)

    # ============================================================
    # Single-image methods
    # ============================================================
    @torch.no_grad()
    def get_bc_fd(self, corp_type: int = -1,
                  fit_range: Optional[Tuple[float, float]] = None):
        if self.img is None:
            raise ValueError("No image was provided to __init__")
        img = (self.img > 0).to(self.dtype)              # 0/1
        scales = self.m_scales
        counts = np.zeros(len(scales), dtype=np.float64)
        it = (tqdm(enumerate(scales), total=len(scales),
                   desc="Calculating by BC (GPU)")
              if self.m_with_progress else enumerate(scales))
        for k, size in it:
            proc = _crop_pad_batch(img.unsqueeze(0), size,
                                    corp_type=corp_type)        # (1,H,W)
            if proc.shape[-2] == 0 or proc.shape[-1] == 0:
                continue
            blocks = _blocks_NHW(proc, size)                    # (1,nb,s*s)
            occ = (blocks.sum(dim=2) > 0).sum(dim=1)            # (1,)
            counts[k] = float(occ.item())
        return _fit_log_log(np.array(scales, dtype=np.float64), counts,
                            fit_range=fit_range)

    @torch.no_grad()
    def get_dbc_fd(self, corp_type: int = -1,
                   fit_range: Optional[Tuple[float, float]] = None):
        if self.img is None:
            raise ValueError("No image was provided to __init__")
        # Honour the user-specified dtype (default fp32). For byte-exact
        # agreement with the CPU class, construct with dtype=torch.float64.
        img = self.img.to(self.dtype)
        H_norm = float(max(img.shape))
        # 99-percentile, fully on-device
        G_max = torch.quantile(img.flatten(), 0.99).clamp_min(1e-12)
        return _single_minmax_fit(
            img, self.m_scales, corp_type=corp_type,
            method="dbc", H_norm=H_norm, G_max=G_max,
            with_progress=self.m_with_progress, fit_range=fit_range,
            device=self.device)

    @torch.no_grad()
    def get_sdbc_fd(self, corp_type: int = -1,
                    fit_range: Optional[Tuple[float, float]] = None):
        if self.img is None:
            raise ValueError("No image was provided to __init__")
        img = self.img.to(self.dtype)
        H_norm = float(max(img.shape))
        G_max = torch.quantile(img.flatten(), 0.99).clamp_min(1e-12)
        return _single_minmax_fit(
            img, self.m_scales, corp_type=corp_type,
            method="sdbc", H_norm=H_norm, G_max=G_max,
            with_progress=self.m_with_progress, fit_range=fit_range,
            device=self.device)

    # ============================================================
    # Batch APIs
    # ============================================================
    @staticmethod
    @torch.no_grad()
    def get_batch_bc(images: Sequence[np.ndarray],
                     max_size: Optional[int] = None,
                     max_scales: int = 30,
                     min_size: int = 2,
                     corp_type: int = -1,
                     fit_range: Optional[Tuple[float, float]] = None,
                     device: Optional[Union[str, torch.device]] = None,
                     dtype: torch.dtype = torch.float32,
                     img_chunk: Optional[int] = None,
                     with_progress: bool = True) -> List[dict]:
        """Batch BC. All images must share the same H, W (binary)."""
        return _batch_run_gpu(images, "bc", corp_type=corp_type,
                              max_size=max_size, max_scales=max_scales,
                              min_size=min_size, fit_range=fit_range,
                              device=device, dtype=dtype,
                              img_chunk=img_chunk,
                              with_progress=with_progress)

    @staticmethod
    @torch.no_grad()
    def get_batch_dbc(images: Sequence[np.ndarray],
                      max_size: Optional[int] = None,
                      max_scales: int = 30,
                      min_size: int = 2,
                      corp_type: int = -1,
                      fit_range: Optional[Tuple[float, float]] = None,
                      device: Optional[Union[str, torch.device]] = None,
                      dtype: torch.dtype = torch.float32,
                      img_chunk: Optional[int] = None,
                      with_progress: bool = True) -> List[dict]:
        """Batch DBC. All images must share the same H, W (grayscale)."""
        return _batch_run_gpu(images, "dbc", corp_type=corp_type,
                              max_size=max_size, max_scales=max_scales,
                              min_size=min_size, fit_range=fit_range,
                              device=device, dtype=dtype,
                              img_chunk=img_chunk,
                              with_progress=with_progress)

    @staticmethod
    @torch.no_grad()
    def get_batch_sdbc(images: Sequence[np.ndarray],
                       max_size: Optional[int] = None,
                       max_scales: int = 30,
                       min_size: int = 2,
                       corp_type: int = -1,
                       fit_range: Optional[Tuple[float, float]] = None,
                       device: Optional[Union[str, torch.device]] = None,
                       dtype: torch.dtype = torch.float32,
                       img_chunk: Optional[int] = None,
                       with_progress: bool = True) -> List[dict]:
        """Batch SDBC. All images must share the same H, W (grayscale)."""
        return _batch_run_gpu(images, "sdbc", corp_type=corp_type,
                              max_size=max_size, max_scales=max_scales,
                              min_size=min_size, fit_range=fit_range,
                              device=device, dtype=dtype,
                              img_chunk=img_chunk,
                              with_progress=with_progress)

    # ============================================================
    # Plot helper
    # ============================================================
    @staticmethod
    def plot(raw_img, gray_img, fd_bc, fd_dbc, fd_sdbc):
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
            r2 = (result["r_value"] ** 2) if result["r_value"] is not None \
                else float("nan")
            scale_range = f"[{min(result['scales'])}, {max(result['scales'])}]"
            plt.plot(x, y, "ro", label="Calculated points", markersize=2)
            plt.plot(x, fd * x + b, "k--", label="Linear fit")
            plt.fill_between(x, fd * x + b - 2 * result["std_err"],
                             fd * x + b + 2 * result["std_err"],
                             color="gray", alpha=0.2,
                             label=r"$\pm 2\sigma$")
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
            plt.title(f"{text}: FD={fd:.4f}", fontsize=7)
            plt.xlabel(r"$\log(1/r)$", fontsize=7)
            plt.ylabel(r"$\log(N(r))$", fontsize=7)
            plt.legend(fontsize=7)
            plt.grid(True, which="both", ls="--", lw=0.3)

        plt.figure(1, figsize=(10, 5))
        plt.subplot(2, 3, 1); show_image("Raw Image", raw_img)
        plt.subplot(2, 3, 3); show_image("Binary Image", gray_img, "gray")
        plt.subplot(2, 3, 4); show_fit("BC", fd_bc)
        plt.subplot(2, 3, 5); show_fit("DBC", fd_dbc)
        plt.subplot(2, 3, 6); show_fit("SDBC", fd_sdbc)
        plt.tight_layout()
        plt.show()


# ============================================================
# Internal helpers
# ============================================================
@torch.no_grad()
def _single_minmax_fit(img: torch.Tensor, scales: List[int],
                       corp_type: int, method: str,
                       H_norm: float, G_max: torch.Tensor,
                       with_progress: bool,
                       fit_range: Optional[Tuple[float, float]],
                       device: torch.device) -> dict:
    """Shared inner loop for DBC / SDBC on a single image."""
    counts = np.zeros(len(scales), dtype=np.float64)
    it = (tqdm(enumerate(scales), total=len(scales),
               desc=f"Calculating by {method.upper()} (GPU)")
          if with_progress else enumerate(scales))
    for k, size in it:
        proc = _crop_pad_batch(img.unsqueeze(0), size,
                                corp_type=corp_type)             # (1,H,W)
        if proc.shape[-2] == 0 or proc.shape[-1] == 0:
            continue
        blocks = _blocks_NHW(proc, size)                         # (1,nb,s*s)
        I_min = blocks.amin(dim=2)                               # (1,nb)
        I_max = blocks.amax(dim=2)                               # (1,nb)
        h_box = float(size) * H_norm / float(G_max.item())
        if method == "dbc":
            n_r = (torch.ceil(I_max / h_box)
                   - torch.ceil(I_min / h_box) + 1.0).clamp_min(1.0)
        else:
            delta = (I_max - I_min).clamp_min(0.0)
            n_r = torch.floor(delta / h_box) + 1.0
        counts[k] = float(n_r.sum().item())
    return _fit_log_log(np.array(scales, dtype=np.float64), counts,
                        fit_range=fit_range)


@torch.no_grad()
def _batch_run_gpu(images: Sequence[np.ndarray], method: str,
                   corp_type: int, max_size: Optional[int],
                   max_scales: int, min_size: int,
                   fit_range: Optional[Tuple[float, float]],
                   device: Optional[Union[str, torch.device]],
                   dtype: torch.dtype,
                   img_chunk: Optional[int],
                   with_progress: bool) -> List[dict]:
    if len(images) == 0:
        return []
    for i, im in enumerate(images):
        if im.ndim != 2:
            raise ValueError(f"image {i} is not 2D")
    shape0 = images[0].shape
    if not all(im.shape == shape0 for im in images):
        raise ValueError(
            "All images must share the same shape for the GPU batch path; "
            "fall back to per-image looping for heterogeneous shapes.")

    N = len(images)
    H_img, W_img = shape0
    if max_size is None:
        max_size = min(H_img, W_img)
    scales = make_scales(int(max_size), max_scales=max_scales,
                         min_size=int(min_size))
    if scales.size < 2:
        raise ValueError(f"Need >= 2 scales; got {scales.size}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if img_chunk is None or img_chunk <= 0:
        img_chunk = N
    img_chunk = min(int(img_chunk), N)

    # Stage all images on host first; we ship to device per-chunk
    # so a giant batch does not blow VRAM in one shot. Use a numpy
    # dtype that matches the requested torch dtype so we do not
    # silently downcast a fp64 user request to fp32.
    np_dtype = np.float64 if dtype == torch.float64 else np.float32
    stack_np = np.stack([im.astype(np_dtype, copy=False)
                         for im in images], axis=0)        # (N, H, W)

    # For DBC / SDBC we need a per-image G_max (99 percentile). Compute
    # it on-device per chunk.
    H_norm = float(max(H_img, W_img))

    counts_all = np.zeros((N, scales.size), dtype=np.float64)

    chunk_iter = (tqdm(range(0, N, img_chunk),
                       desc=f"Batch {method.upper()} (GPU)")
                  if with_progress else range(0, N, img_chunk))

    for lo in chunk_iter:
        hi = min(N, lo + img_chunk)
        b = hi - lo
        chunk_t = torch.from_numpy(stack_np[lo:hi]).to(
            device=device, dtype=dtype)                  # (b, H, W)

        if method == "bc":
            occ_chunk = (chunk_t > 0).to(dtype)          # 0/1, (b, H, W)
            for k, size in enumerate(scales):
                size = int(size)
                proc = _crop_pad_batch(occ_chunk, size, corp_type=corp_type)
                if proc.shape[-2] == 0 or proc.shape[-1] == 0:
                    continue
                blocks = _blocks_NHW(proc, size)                  # (b,nb,s*s)
                occ = (blocks.sum(dim=2) > 0).sum(dim=1)          # (b,)
                counts_all[lo:hi, k] = occ.detach().cpu().numpy()
            del occ_chunk
        else:
            # DBC / SDBC: per-image 99-percentile, on-device
            flat = chunk_t.reshape(b, -1)
            try:
                G_max = torch.quantile(flat, 0.99, dim=1)         # (b,)
            except RuntimeError:
                # Some torch builds limit quantile input size; fall back
                # to a chunked approximation via topk.
                k_top = max(1, int(round(0.01 * flat.shape[1])))
                G_max = torch.topk(flat, k_top, dim=1).values.amin(dim=1)
            G_max = G_max.clamp_min(1e-12)                        # (b,)
            for k, size in enumerate(scales):
                size = int(size)
                proc = _crop_pad_batch(chunk_t, size, corp_type=corp_type)
                if proc.shape[-2] == 0 or proc.shape[-1] == 0:
                    continue
                blocks = _blocks_NHW(proc, size)                  # (b,nb,s*s)
                I_min = blocks.amin(dim=2)                        # (b,nb)
                I_max = blocks.amax(dim=2)                        # (b,nb)
                h_box = (float(size) * H_norm
                         / G_max).reshape(b, 1)                   # (b, 1)
                if method == "dbc":
                    n_r = (torch.ceil(I_max / h_box)
                           - torch.ceil(I_min / h_box) + 1.0
                           ).clamp_min(1.0)
                else:
                    delta = (I_max - I_min).clamp_min(0.0)
                    n_r = torch.floor(delta / h_box) + 1.0
                counts_all[lo:hi, k] = (n_r.sum(dim=1)
                                        .detach().cpu().numpy())

        del chunk_t
        if device.type == "cuda":
            torch.cuda.empty_cache()

    scale_arr = scales.astype(np.float64)
    return [_fit_log_log(scale_arr, counts_all[i], fit_range=fit_range)
            for i in range(N)]


# ============================================================
# Demo
# ============================================================
def main():
    import cv2, time, os
    image_path = "../images/fractal.png"
    if not os.path.exists(image_path):
        # Synthetic Sierpinski carpet (theory FD = log(8)/log(3))
        size = 729
        img = np.ones((size, size), dtype=np.uint8) * 255
        def carve(x0, y0, n):
            if n < 1:
                return
            t = 3 ** (n - 1)
            img[y0 + t:y0 + 2 * t, x0 + t:x0 + 2 * t] = 0
            for dy in range(3):
                for dx in range(3):
                    if dx == 1 and dy == 1:
                        continue
                    carve(x0 + dx * t, y0 + dy * t, n - 1)
        carve(0, 0, 6)
        raw_image = img
    else:
        raw_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    raw_image = raw_image.astype(np.float32)
    bin_image = (raw_image < 64).astype(np.uint8)

    # ---- single image ----
    t0 = time.time()
    fd_bc = CFAImageFDGPU(bin_image, max_scales=30,
                                 with_progress=False).get_bc_fd(corp_type=-1)
    fd_dbc = CFAImageFDGPU(raw_image, max_scales=30,
                                  with_progress=False).get_dbc_fd(corp_type=-1)
    fd_sdbc = CFAImageFDGPU(raw_image, max_scales=30,
                                   with_progress=False).get_sdbc_fd(corp_type=-1)
    print(f"Single image: {time.time()-t0:.3f}s")
    print(f"  BC FD   = {fd_bc['fd']:.4f}")
    print(f"  DBC FD  = {fd_dbc['fd']:.4f}")
    print(f"  SDBC FD = {fd_sdbc['fd']:.4f}")

    # ---- batch ----
    bin_imgs = [bin_image] * 4
    gray_imgs = [raw_image] * 4
    t0 = time.time()
    bc_list = CFAImageFDGPU.get_batch_bc(bin_imgs, max_scales=30,
                                                with_progress=False)
    dbc_list = CFAImageFDGPU.get_batch_dbc(gray_imgs, max_scales=30,
                                                   with_progress=False)
    sdbc_list = CFAImageFDGPU.get_batch_sdbc(gray_imgs, max_scales=30,
                                                     with_progress=False)
    print(f"Batch (4 imgs): {time.time()-t0:.3f}s")
    print(f"  batch BC FD[0]   = {bc_list[0]['fd']:.4f}")
    print(f"  batch DBC FD[0]  = {dbc_list[0]['fd']:.4f}")
    print(f"  batch SDBC FD[0] = {sdbc_list[0]['fd']:.4f}")


if __name__ == "__main__":
    main()
