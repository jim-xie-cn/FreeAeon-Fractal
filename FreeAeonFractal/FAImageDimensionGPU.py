import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm

class CFAImageDimensionGPU:
    """
    GPU-accelerated fractal dimension calculations (BC / DBC / SDBC) using PyTorch.
    image: 2D numpy array (H,W), grayscale or binary
    """

    def __init__(self, image=None, max_size=None, max_scales=100, with_progress=True, device=None):
        assert image is not None and image.ndim == 2, "image must be a 2D single-channel array"
        self.m_with_progress = with_progress

        if max_size is None:
            max_size = min(image.shape)

        scales = np.logspace(1, np.log2(max_size), num=max_scales, base=2, dtype=int)
        scales = np.unique(scales)
        self.m_scales = [int(s) for s in scales if s > 0]

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # keep image on GPU
        img_t = torch.from_numpy(np.asarray(image)).to(self.device)
        if img_t.dtype not in (torch.float16, torch.float32, torch.float64):
            img_t = img_t.float()
        else:
            img_t = img_t.float()
        self.img = img_t  # (H,W)

    # -------------------------
    # utilities
    # -------------------------
    @staticmethod
    def _linregress_from_logs(log_scales_np, log_counts_np):
        """
        Replace scipy.stats.linregress to avoid CPU scipy dependency.
        Works on numpy arrays.
        Returns slope, intercept, r_value, p_value(None), std_err.
        """
        x = log_scales_np.astype(np.float64)
        y = log_counts_np.astype(np.float64)
        n = x.size
        x_mean = x.mean()
        y_mean = y.mean()

        ss_xx = np.sum((x - x_mean) ** 2)
        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        slope = ss_xy / (ss_xx + 1e-12)
        intercept = y_mean - slope * x_mean

        # r_value
        ss_yy = np.sum((y - y_mean) ** 2)
        r_num = ss_xy
        r_den = math.sqrt((ss_xx + 1e-12) * (ss_yy + 1e-12))
        r_value = r_num / (r_den + 1e-12)

        # standard error of slope
        y_hat = slope * x + intercept
        residual = y - y_hat
        s_err = math.sqrt(np.sum(residual ** 2) / max(n - 2, 1))
        std_err = s_err / math.sqrt(ss_xx + 1e-12)

        return slope, intercept, r_value, None, std_err

    def get_fd(self, scale_list, box_count_list):
        s_list = np.array(scale_list, dtype=np.float64)
        b_list = np.array(box_count_list, dtype=np.float64)
        b_list = np.where(b_list == 0, np.finfo(float).eps, b_list)
        s_list = np.where(s_list == 0, np.finfo(float).eps, s_list)

        log_scales = -np.log(s_list)
        log_counts = np.log(b_list)

        slope, intercept, r_value, p_value, std_err = self._linregress_from_logs(log_scales, log_counts)

        return {
            "fd": float(slope),
            "scales": s_list.tolist(),
            "counts": b_list.tolist(),
            "log_scales": log_scales.tolist(),
            "log_counts": log_counts.tolist(),
            "intercept": float(intercept),
            "r_value": float(r_value),
            "p_value": p_value,        # 未计算；若必须要p值，可再接scipy或自己补t检验
            "std_err": float(std_err),
        }

    def _crop_or_pad_to_multiple(self, img2d, size, corp_type=-1):
        """
        corp_type:
          -1: crop to multiple of size
           0: no processing (要求刚好整除，否则会报错)
           1: pad to multiple of size (zero-pad)
        returns: processed image, newH, newW
        """
        H, W = img2d.shape
        if corp_type == 0:
            if (H % size) != 0 or (W % size) != 0:
                raise ValueError(f"corp_type=0 requires H,W divisible by size. Got {(H,W)} size={size}")
            return img2d, H, W

        if corp_type == -1:
            newH = (H // size) * size
            newW = (W // size) * size
            return img2d[:newH, :newW], newH, newW

        if corp_type == 1:
            newH = ((H + size - 1) // size) * size
            newW = ((W + size - 1) // size) * size
            padH = newH - H
            padW = newW - W
            # pad format for 2D: (left,right,top,bottom)
            img_pad = F.pad(img2d.unsqueeze(0).unsqueeze(0), (0, padW, 0, padH), mode="constant", value=0.0)
            return img_pad[0, 0], newH, newW

        raise ValueError("corp_type must be -1,0,1")

    def _view_as_blocks(self, img2d, size, corp_type=-1):
        """
        returns blocks as (nb, size*size) on GPU
        """
        img2d, H, W = self._crop_or_pad_to_multiple(img2d, size, corp_type=corp_type)
        # (H,W) -> (H/size, size, W/size, size) -> (nb, size*size)
        h = H // size
        w = W // size
        blocks = img2d.reshape(h, size, w, size).permute(0, 2, 1, 3).reshape(h * w, size * size)
        return blocks

    # -------------------------
    # BC / DBC / SDBC on GPU
    # -------------------------
    @torch.no_grad()
    def get_bc_fd(self, corp_type=-1):
        """
        BC: count non-empty boxes in binary image.
        Expect image is 0/1 or 0/255; any positive treated as occupied.
        """
        scale_list, box_count_list = [], []
        it = tqdm(self.m_scales, desc="Calculating by BC (GPU)") if self.m_with_progress else self.m_scales

        img = self.img
        # ensure binary-ish: >0 as occupied
        img_occ = (img > 0).to(torch.int32)

        for size in it:
            blocks = self._view_as_blocks(img_occ, size, corp_type=corp_type)  # (nb, s*s)
            occ = (blocks.sum(dim=1) > 0).sum().item()
            scale_list.append(size)
            box_count_list.append(int(occ))

        return self.get_fd(scale_list, box_count_list)

    @torch.no_grad()
    def get_dbc_fd(self, corp_type=-1):
        """
        DBC: for each box compute min/max gray, map to height Hnorm, count layers.
        """
        scale_list, box_count_list = [], []
        it = tqdm(self.m_scales, desc="Calculating by DBC (GPU)") if self.m_with_progress else self.m_scales

        img = self.img
        Hnorm = float(max(img.shape))  # same as original

        # percentile on GPU has a potential RuntimeError: quantile() input tensor is too large, use CPU here
        # gray_max = torch.quantile(img.flatten(), 0.99).clamp_min(1e-12)
        gray_max = torch.tensor(np.quantile(img.cpu().numpy(), 0.99),device=self.device, dtype=torch.float32).clamp_min(1e-12)

        for size in it:
            blocks = self._view_as_blocks(img, size, corp_type=corp_type)  # (nb, s*s)
            I_min = blocks.min(dim=1).values
            I_max = blocks.max(dim=1).values

            Z_min = (I_min / gray_max) * Hnorm
            Z_max = (I_max / gray_max) * Hnorm
            delta_z = (Z_max - Z_min).clamp_min(0.0)

            box_cnt = torch.ceil((delta_z + 1e-6) / float(size)).to(torch.int64).sum().item()
            scale_list.append(size)
            box_count_list.append(int(box_cnt))

        return self.get_fd(scale_list, box_count_list)

    @torch.no_grad()
    def get_sdbc_fd(self, corp_type=-1):
        """
        use dbc as place holder
        """
        return self.get_dbc_fd(corp_type=corp_type)

    @torch.no_grad()
    def get_batch_bc_fd(self, images, corp_type=-1):
        """
        calaculate fd with box counting
        images: numpy array or torch tensor, shape = (N, H, W)
        corp_type: -1:corp, 1:pad, 0:no corp/pad
        return: FD list
        """

        if isinstance(images, np.ndarray):
            images_t = torch.from_numpy(images).to(self.device)
        elif isinstance(images, torch.Tensor):
            images_t = images.to(self.device)
        else:
            raise TypeError("images must be numpy array or torch tensor")

        if images_t.dtype not in (torch.float16, torch.float32, torch.float64):
            images_t = images_t.float()
        else:
            images_t = images_t.float()

        img_occ = (images_t > 0).to(torch.int32)  # (N,H,W)

        N = img_occ.shape[0]
        scale_results = [[] for _ in range(N)]
        count_results = [[] for _ in range(N)]

        it = tqdm(self.m_scales, desc="Batch BC (GPU)") if self.m_with_progress else self.m_scales

        for size in it:
            processed_imgs = []
            new_HW_list = []
            for idx in range(N):
                img_proc, newH, newW = self._crop_or_pad_to_multiple(img_occ[idx], size, corp_type=corp_type)
                processed_imgs.append(img_proc)
                new_HW_list.append((newH, newW))

            batch_imgs = torch.stack(processed_imgs)  # (N,H_new,W_new)

            H_new, W_new = new_HW_list[0]
            h = H_new // size
            w = W_new // size
            blocks = batch_imgs.reshape(N, h, size, w, size) \
                .permute(0, 1, 3, 2, 4) \
                .reshape(N, h * w, size * size)  # (N, nb, s*s)

            occ = (blocks.sum(dim=2) > 0).sum(dim=1)  

            for idx in range(N):
                scale_results[idx].append(size)
                count_results[idx].append(int(occ[idx].item()))

        return [self.get_fd(scale_results[idx], count_results[idx]) for idx in range(N)]

    @torch.no_grad()
    def get_batch_dbc_fd(self, images, corp_type=-1):
        """
        batch calcute FD with DBC
        images: numpy array or torch tensor, shape = (N, H, W)
        corp_type: -1:corp, 1:pad, 0:no corp/pad
        return: FD list of images, 
        """

        if isinstance(images, np.ndarray):
            images_t = torch.from_numpy(images).to(self.device)
        elif isinstance(images, torch.Tensor):
            images_t = images.to(self.device)
        else:
            raise TypeError("images must be numpy array or torch tensor")

        if images_t.dtype not in (torch.float16, torch.float32, torch.float64):
            images_t = images_t.float()
        else:
            images_t = images_t.float()

        N, H, W = images_t.shape
        Hnorm = float(max(H, W))

        gray_max_list = torch.tensor(
            np.quantile(images_t.cpu().numpy(), 0.99, axis=(1, 2)),
            device=self.device, dtype=torch.float32
        ).clamp_min(1e-12)

        scale_results = [[] for _ in range(N)]
        count_results = [[] for _ in range(N)]

        it = tqdm(self.m_scales, desc="Batch DBC (GPU)") if self.m_with_progress else self.m_scales

        for size in it:
            processed_imgs = []
            new_HW_list = []
            for idx in range(N):
                img_proc, newH, newW = self._crop_or_pad_to_multiple(images_t[idx], size, corp_type=corp_type)
                processed_imgs.append(img_proc)
                new_HW_list.append((newH, newW))

            batch_imgs = torch.stack(processed_imgs)  # (N, newH, newW)

            H_new, W_new = new_HW_list[0]
            h = H_new // size
            w = W_new // size
            blocks = batch_imgs.reshape(N, h, size, w, size) \
                .permute(0, 1, 3, 2, 4) \
                .reshape(N, h * w, size * size)  # (N, nb, size*size)

            # min/max
            I_min = blocks.min(dim=2).values   # (N, nb)
            I_max = blocks.max(dim=2).values   # (N, nb)

            Z_min = (I_min / gray_max_list.unsqueeze(1)) * Hnorm
            Z_max = (I_max / gray_max_list.unsqueeze(1)) * Hnorm
            delta_z = (Z_max - Z_min).clamp_min(0.0)

            box_cnt = torch.ceil((delta_z + 1e-6) / float(size)).to(torch.int64).sum(dim=1)  # (N,)

            for idx in range(N):
                scale_results[idx].append(size)
                count_results[idx].append(int(box_cnt[idx].item()))

        return [self.get_fd(scale_results[idx], count_results[idx]) for idx in range(N)]

    '''get batch DBC FD'''
    @staticmethod
    def get_batch_dbc(img_list, max_scales=100, with_progress=True, device=None, corp_type=-1):
        imgs_np = np.stack(img_list, axis=0)  # (N,H,W)
        fd_list = CFAImageDimensionGPU(imgs_np[0],max_scales=max_scales, with_progress=with_progress,device=device).get_batch_dbc_fd(imgs_np, corp_type=corp_type)
        return fd_list

    '''get batch BC FD'''
    @staticmethod
    def get_batch_bc(img_list, max_scales=100, with_progress=True, device=None, corp_type=-1):
        imgs_np = np.stack(img_list, axis=0)  # (N,H,W)
        fd_list = CFAImageDimensionGPU(imgs_np[0],max_scales=max_scales, with_progress=with_progress,device=device).get_batch_bc_fd(imgs_np, corp_type=corp_type)
        return fd_list

    '''Display image and fitting plots for various FD calculations'''
    @staticmethod
    def plot(raw_img, gray_img, fd_bc, fd_dbc, fd_sdbc):
        def show_image(text,image,cmap='viridis'):
            plt.imshow(image, cmap=cmap)
            plt.title(text,fontsize=8)
            plt.axis('off')
        def show_fit(text,result):
            #x = np.array(result['log_scales'])
            #y = np.array(result['log_counts'])
            #fd = result['fd']
            #b = result['intercept']
            #plt.title('%s: FD=%0.4f PV=%.4f' % (text,fd,result['p_value']),fontsize=8)
            #b = result['intercept']
            #plt.plot(x, y, 'ro',label='Calculated points',markersize=1)
            #plt.plot(x, fd*x+b, 'k--', label='Linear fit')
            #plt.legend(loc=4,fontsize=8)
            #plt.xlabel('$log(1/r)$',fontsize=8)
            #plt.ylabel('$log(Nr)$',fontsize=8)
            #plt.legend(fontsize=8)
            x = np.array(result['log_scales'])
            y = np.array(result['log_counts'])
            fd = result['fd']
            b = result['intercept']
            r2 = result['r_value'] ** 2
            scale_range = f"[{min(result['scales'])}, {max(result['scales'])}]"

            plt.plot(x, y, 'ro', label='Calculated points', markersize=2)
            plt.plot(x, fd * x + b, 'k--', label='Linear fit')
            plt.fill_between(x, fd*x + b - 2*result['std_err'], fd*x + b + 2*result['std_err'],
                 color='gray', alpha=0.2, label='±2σ band')

            textstr = '\n'.join((r'$D=%.4f$' % (fd,), r'$R^2=%.4f$' % (r2,),r'Scale: ' + scale_range))

            plt.gca().text(0.95, 0.95, textstr, transform=plt.gca().transAxes,fontsize=7, verticalalignment='top', horizontalalignment='right',bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.5))
            plt.title('%s: FD=%0.4f' % (text,fd),fontsize=7)
            plt.xlabel(r'$\log(1/r)$', fontsize=7)
            plt.ylabel(r'$\log(N(r))$', fontsize=7)
            plt.legend(fontsize=7)
            plt.grid(True, which='both', ls='--', lw=0.3)

        plt.figure(1,figsize=(10,5))
        plt.subplot(2, 3, 1)
        show_image("Raw Image",raw_img)
        plt.subplot(2, 3, 3)
        show_image("Binary Image",gray_img,"gray")
        plt.subplot(2, 3, 4)
        show_fit("BC",fd_bc)
        plt.subplot(2, 3, 5)
        show_fit("DBC",fd_dbc)
        plt.subplot(2, 3, 6)
        show_fit("SDBC",fd_sdbc)

        plt.tight_layout()
        plt.show()

def main():
    import cv2,time
    max_scales = 10240
    raw_image = cv2.imread("../images/fractal.png", cv2.IMREAD_GRAYSCALE)
    raw_image = raw_image.astype(np.float32)
    bin_image = (raw_image < 64).astype(np.uint8)
    
    start = time.time()
    for i in tqdm(range(2)):
        fd_bc   = CFAImageDimensionGPU(bin_image,max_scales=max_scales, with_progress=False).get_bc_fd(corp_type=-1)
        fd_dbc  = CFAImageDimensionGPU(raw_image,max_scales=max_scales, with_progress=False).get_dbc_fd(corp_type=-1)
    
    print("Time used (No batch) ",time.time() - start)
    print("BC FD:", fd_bc["fd"])
    print("DBC FD:", fd_dbc["fd"])
    
    gray_imgs = []
    for i in range(2):
        gray_imgs.append(raw_image)
    bin_imgs = []
    for i in range(2):
        bin_imgs.append(bin_image)
    
    start = time.time()
    bc_fd_list = CFAImageDimensionGPU.get_batch_bc(bin_imgs,with_progress=False, corp_type=-1,max_scales=max_scales)
    dbc_fd_list = CFAImageDimensionGPU.get_batch_dbc(gray_imgs,with_progress=False, corp_type=-1,max_scales=max_scales)
    
    print("Time used (batch) ",time.time() - start)
    print("BC FD:", bc_fd_list[0]['fd'])
    print("DBC FD:", dbc_fd_list[0]['fd'])

if __name__ == "__main__":
    main()

