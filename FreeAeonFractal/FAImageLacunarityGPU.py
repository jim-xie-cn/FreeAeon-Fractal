import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tqdm import tqdm

class CFAImage:
    @staticmethod
    def otsu_binarize(img):
        _, thr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return thr, _

class CFAImageLacunarity:
    def __init__(self, image=None, max_size=None, max_scales=100,
                 with_progress=True, scales_mode="powers", partition_mode="gliding"):
        if image is None:
            raise ValueError("Image must be provided.")
        self.image = image
        self.with_progress = with_progress
        self.partition_mode = partition_mode.lower()

        if max_size is None:
            max_size = min(image.shape)

        if scales_mode == "powers":
            max_pow = int(np.log2(max_size))
            self.scales = list(2 ** np.arange(1, max_pow + 1))
        elif scales_mode == "logspace":
            scales = np.logspace(0, np.log2(max_size), num=max_scales, base=2)
            scales = np.unique(np.round(scales).astype(int))
            scales = scales[scales <= max_size]
            self.scales = [int(s) for s in scales if s > 0]

        self.max_size = max_size
        self.max_scales = max_scales
        self.scales_mode = scales_mode

    @staticmethod
    def _safe_lacunarity_from_masses(masses, eps=1e-12):
        if isinstance(masses, torch.Tensor):
            if masses.numel() == 0:
                return torch.tensor(float("nan"), device=masses.device)
            mean_m = masses.mean()
            if mean_m <= eps:
                return torch.tensor(float("nan"), device=masses.device)
            mean_m2 = (masses*masses).mean()
            return mean_m2/(mean_m*mean_m)
        m = np.asarray(masses, dtype=np.float64)
        if m.size == 0:
            return np.nan
        mean_m = m.mean()
        if mean_m <= eps:
            return np.nan
        mean_m2 = (m*m).mean()
        return mean_m2/(mean_m*mean_m)

    @staticmethod
    def _gpu_lacunarity(imgs_t, scales, partition_mode, include_zero, with_progress):
        dtype = imgs_t.dtype
        device = imgs_t.device
        B,H,W = imgs_t.shape
        results = [[] for _ in range(B)]
        iterator = tqdm(scales, desc="GPU Lacunarity", disable=not with_progress)
        for size in iterator:
            if size > H or size > W:
                for i in range(B):
                    results[i].append({
                        "scale": int(size),
                         "num_boxes": 0,
                        "mean_mass": np.nan,
                        "var_mass": np.nan,
                        "lambda": np.nan
                    })
                continue
            if partition_mode == "non-overlapping":
                nY = H//size
                nX = W//size
                blocks = imgs_t.reshape(B, nY, size, nX, size)
                masses = blocks.sum(dim=(2,4)).reshape(B,-1)
            elif partition_mode == "gliding":
                S = torch.zeros((B,H+1,W+1), device=device, dtype=dtype)
                S[:,1:,1:] = torch.cumsum(torch.cumsum(imgs_t,dim=1),dim=2)
                y1 = torch.arange(size,H+1,device=device)
                x1 = torch.arange(size,W+1,device=device)
                Y1,X1 = torch.meshgrid(y1,x1,indexing="ij")
                Y0 = Y1-size
                X0 = X1-size
                masses = S[:,Y1,X1] - S[:,Y0,X1] - S[:,Y1,X0] + S[:,Y0,X0]
                masses = masses.reshape(B,-1)
            if not include_zero:
                masses = torch.stack([m[m>0] for m in masses])
            for i in range(B):
                lambda_val = CFAImageLacunarity._safe_lacunarity_from_masses(masses[i])
                results[i].append({
                    "scale": int(size),
                    "num_boxes": int(masses[i].numel()),
                    "mean_mass": float(masses[i].mean().item()) if masses[i].numel()>0 else np.nan,
                    "var_mass": float(masses[i].var(unbiased=False).item()) if masses[i].numel()>0 else np.nan,
                    "lambda": float(lambda_val.item()) if isinstance(lambda_val, torch.Tensor) else lambda_val
                })
        return results

    def get_lacunarity(self, use_binary_mass=False, include_zero=True, device=None,corp_type=-1):
        device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = torch.float64
        img_t = torch.tensor(self.image.astype(np.float64), device=device, dtype=dtype)
        if use_binary_mass:
            img_t = (img_t > 0).to(dtype)
        res_list = self._gpu_lacunarity(
            img_t.unsqueeze(0),
            self.scales,
            self.partition_mode,
            include_zero,
            self.with_progress
        )
        r = res_list[0]
        return {
            "scales": [e["scale"] for e in r],
            "lacunarity": [e["lambda"] for e in r],
            "mass_stats": r
        }

    @staticmethod
    def get_batch_lacunarity(img_list, scales=None, scales_mode="logspace",
                             max_size=None, max_scales=100,
                             partition_mode="gliding", use_binary_mass=False,
                             include_zero=True, device=None, with_progress=True):
        if scales is None:
            if max_size is None:
                max_size = min(min(img.shape) for img in img_list)
            if scales_mode == "powers":
                max_pow = int(np.log2(max_size))
                scales = list(2 ** np.arange(1, max_pow + 1))
            elif scales_mode == "logspace":
                scales = np.logspace(0, np.log2(max_size), num=max_scales, base=2)
                scales = np.unique(np.round(scales).astype(int))
                scales = scales[scales <= max_size]
                scales = [int(s) for s in scales if s > 0]

        device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = torch.float64
        imgs_t = []
        for img in img_list:
            arr = torch.tensor(img.astype(np.float64), device=device, dtype=dtype)
            if use_binary_mass:
                arr = (arr > 0).to(dtype)
            imgs_t.append(arr)
        imgs_t = torch.stack(imgs_t)
        res_list = CFAImageLacunarity._gpu_lacunarity(
            imgs_t,
            scales,
            partition_mode,
            include_zero,
            with_progress
        )
        return [{
            "scales": [e["scale"] for e in r],
            "lacunarity": [e["lambda"] for e in r],
            "mass_stats": r
        } for r in res_list]

    @staticmethod
    def fit_lacunarity(lac_result):
        scales = np.asarray(lac_result["scales"], dtype=np.float64)
        lac_vals = np.asarray(lac_result["lacunarity"], dtype=np.float64)
        mask = np.isfinite(scales) & (scales>0) & np.isfinite(lac_vals) & (lac_vals>1)
        x = np.log(scales[mask])
        y = np.log(lac_vals[mask] - 1.0)
        if x.size < 2:
            return dict(slope=np.nan)
        slope, intercept, r_value, p_value, std_err = linregress(x,y)
        return dict(slope=float(slope), r_value=float(r_value))

    @staticmethod
    def plot(lac_result, title="Lacunarity"):
        plt.figure(figsize=(6,4))
        plt.plot(lac_result["scales"], lac_result["lacunarity"], 'o-', lw=1.5)
        plt.xlabel("Scale")
        plt.ylabel("Lacunarity")
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    img = cv2.imread("../images/face.png", cv2.IMREAD_GRAYSCALE)

    bin_img, _ = CFAImage.otsu_binarize(img)

    calc = CFAImageLacunarity(bin_img, scales_mode="powers", partition_mode="gliding")
    single_res = calc.get_lacunarity(use_binary_mass=True)
    single_fit = CFAImageLacunarity.fit_lacunarity(single_res)
    print("Single result:", single_res)
    print("Single slope:", single_fit)

    imgs = [bin_img, bin_img]
    batch_res = CFAImageLacunarity.get_batch_lacunarity(
        imgs,
        scales=None, 
        scales_mode="powers",
        partition_mode="gliding",
        use_binary_mass=True,
        device="cuda"
    )

    print("\nBatch results:")
    for i, res in enumerate(batch_res):
        fit = CFAImageLacunarity.fit_lacunarity(res)
        print(f"Image {i} slope:", fit)

    CFAImageLacunarity.plot(single_res, title="Single Image Lacunarity")
    CFAImageLacunarity.plot(batch_res[0], title="Batch Image 0 Lacunarity")

