"""
Multifractal utilities: image blocking/merging + ROI extraction by q.

Features
- Supports 2D grayscale (H,W) and 3D color (H,W,C).
- Block operations only partition spatial dimensions (H, W); channels are preserved.
- ROI by q uses a probability measure mu per box:
    mu_i = box_sum / total_sum on the CROPPED region.
- Robust handling for q<0 by excluding zero-mass boxes.
- For any q, boxes with mu==0 will NEVER get positive weight (prevents background selection bugs).

Requirements:
    numpy, opencv-python, matplotlib, scikit-image
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.util import view_as_blocks


class CFAImage(object):
    # -----------------------------
    # Crop / Pad
    # -----------------------------
    @staticmethod
    def crop_data(data, block_size):
        """Crop spatial dimensions so H and W are multiples of block_size."""
        if data is None:
            raise ValueError("data is None")
        if len(block_size) != 2:
            raise ValueError("block_size must be (bh, bw)")

        h, w = data.shape[:2]
        bh, bw = block_size
        new_h = (h // bh) * bh
        new_w = (w // bw) * bw

        if data.ndim == 2:
            return data[:new_h, :new_w]
        elif data.ndim == 3:
            return data[:new_h, :new_w, :]
        else:
            raise ValueError("Unsupported data ndim (only 2D/3D supported)")

    @staticmethod
    def pad_data(data, block_size, mode="constant", constant_values=0):
        """Pad spatial dimensions so H and W are multiples of block_size."""
        if data is None:
            raise ValueError("data is None")
        if len(block_size) != 2:
            raise ValueError("block_size must be (bh, bw)")

        h, w = data.shape[:2]
        bh, bw = block_size

        pad_h = (bh - (h % bh)) % bh
        pad_w = (bw - (w % bw)) % bw

        if data.ndim == 2:
            pad_width = ((0, pad_h), (0, pad_w))
        elif data.ndim == 3:
            pad_width = ((0, pad_h), (0, pad_w), (0, 0))
        else:
            raise ValueError("Unsupported data ndim (only 2D/3D supported)")

        # Note: constant_values is only meaningful for mode='constant'
        if mode == "constant":
            return np.pad(data, pad_width, mode=mode, constant_values=constant_values)
        return np.pad(data, pad_width, mode=mode)

    # -----------------------------
    # Block split / merge (spatial only)
    # -----------------------------
    @staticmethod
    def get_boxes_from_image(image, block_size, corp_type=-1):
        """
        Split an image into blocks of size (bh, bw) over spatial dimensions.
        Returns:
            blocks_reshaped:
                grayscale: (num_blocks, bh, bw)
                color:     (num_blocks, bh, bw, c)
            raw_blocks (grid layout):
                grayscale: (nY, nX, bh, bw)
                color:     (nY, nX, bh, bw, c)
        """
        if image is None:
            raise ValueError("image is None")
        if len(block_size) != 2:
            raise ValueError("block_size must be (bh, bw)")

        if corp_type == -1:
            corp_data = CFAImage.crop_data(image, block_size)
        elif corp_type == 1:
            corp_data = CFAImage.pad_data(image, block_size)
        else:
            corp_data = image

        bh, bw = block_size

        if corp_data.ndim == 2:
            raw_blocks = view_as_blocks(corp_data, block_shape=(bh, bw))  # (nY, nX, bh, bw)
            nY, nX = raw_blocks.shape[:2]
            blocks_reshaped = raw_blocks.reshape(nY * nX, bh, bw)
            return blocks_reshaped, raw_blocks

        if corp_data.ndim == 3:
            c = corp_data.shape[2]
            rb = view_as_blocks(corp_data, block_shape=(bh, bw, c))
            raw_blocks = CFAImage._normalize_color_raw_blocks(rb, bh, bw, c)  # -> (nY, nX, bh, bw, c)
            nY, nX = raw_blocks.shape[:2]
            blocks_reshaped = raw_blocks.reshape(nY * nX, bh, bw, c)
            return blocks_reshaped, raw_blocks

        raise ValueError("Unsupported image ndim (only 2D/3D supported)")

    @staticmethod
    def _normalize_color_raw_blocks(raw_blocks, bh, bw, c):
        """
        Normalize view_as_blocks output for color images to shape (nY, nX, bh, bw, c).

        skimage versions can output shapes like:
            - (nY, nX, bh, bw, c)
            - (nY, nX, 1, bh, bw, c)
            - (nY, nX, 1, 1, 1, bh, bw, c)  (rare)
        Strategy:
            - Ensure last 3 dims are (bh, bw, c)
            - Squeeze any singleton dims between (nY,nX) and (bh,bw,c)
        """
        rb = raw_blocks
        if rb.ndim < 5:
            raise ValueError(f"Unexpected raw_blocks ndim for color image: {rb.ndim}, shape={rb.shape}")

        if tuple(rb.shape[-3:]) != (bh, bw, c):
            # Some layouts already are (nY,nX,bh,bw,c); then last 3 would be (bw,c) only.
            # Try detect that case:
            if rb.ndim == 5 and tuple(rb.shape[2:]) == (bh, bw, c):
                return rb
            raise ValueError(f"Unexpected raw_blocks shape tail for color image: {rb.shape}, expected ...,(bh,bw,c)")

        # Now last 3 dims are correct. We want first two dims as (nY,nX).
        # Squeeze singleton dims between them.
        nY, nX = rb.shape[0], rb.shape[1]
        mid = rb.shape[2:-3]
        if all(m == 1 for m in mid):
            rb2 = rb.reshape((nY, nX, bh, bw, c))
            return rb2

        # If there are non-singleton mid dims, it's an unknown layout
        raise ValueError(f"Unexpected raw_blocks middle dims for color image: shape={rb.shape}")

    @staticmethod
    def get_image_from_boxes(raw_blocks):
        """
        Merge raw_blocks back into an image.
        Supports:
            grayscale raw_blocks: (nY, nX, bh, bw) -> (H, W)
            color raw_blocks:     (nY, nX, bh, bw, c) -> (H, W, c)
        """
        if raw_blocks is None:
            raise ValueError("raw_blocks is None")

        if raw_blocks.ndim == 4:
            nY, nX, bh, bw = raw_blocks.shape
            return raw_blocks.transpose(0, 2, 1, 3).reshape(nY * bh, nX * bw)

        if raw_blocks.ndim == 5:
            nY, nX, bh, bw, c = raw_blocks.shape
            return raw_blocks.transpose(0, 2, 1, 3, 4).reshape(nY * bh, nX * bw, c)

        raise ValueError("Unsupported raw_blocks ndim (only 4/5 supported)")

    @staticmethod
    def get_mask_from_boxes(raw_blocks, mask_block_pos):
        """
        Build a binary mask image where blocks in mask_block_pos are set to 0, others 1.
        mask_block_pos: list of (y, x) indices in block-grid coordinates.

        Returns:
            mask_img: (H, W) float32 in {0,1}
        """
        if raw_blocks is None:
            raise ValueError("raw_blocks is None")

        mask_set = set(map(tuple, mask_block_pos))

        if raw_blocks.ndim == 4:
            nY, nX, bh, bw = raw_blocks.shape
        elif raw_blocks.ndim == 5:
            nY, nX, bh, bw = raw_blocks.shape[:4]
        else:
            raise ValueError("Unsupported raw_blocks ndim (only 4/5 supported)")

        mask_blocks = np.ones((nY, nX, bh, bw), dtype=np.float32)
        for y in range(nY):
            for x in range(nX):
                if (y, x) in mask_set:
                    mask_blocks[y, x, :, :] = 0.0

        return CFAImage.get_image_from_boxes(mask_blocks)

    # -----------------------------
    # ROI extraction by q
    # -----------------------------
    @staticmethod
    def get_roi_by_q(
        image,
        q_range=(-5, 5),
        step=1.0,
        box_size=16,
        target_mass=0.95,
        combine_mode="and",
        use_grayscale_measure=True,
        measure_mode="intensity_sum",
    ):
        """
        Extract ROI by selecting boxes with top cumulative mass under weights proportional to mu^q.

        Steps:
        1) Crop to multiples of box_size.
        2) Compute per-box measure m_i (default: sum of pixel intensities in the box).
        3) Normalize mu_i = m_i / sum(m_i).
        4) For each q:
           - active boxes are those with mu_i > 0 (ALWAYS).
           - weights w_i = exp(q * log(mu_i)) on active boxes; zero otherwise.
           - normalize p_i = w_i / sum(w_i).
           - select largest boxes until cumulative sum >= target_mass.
        5) OR masks across q (per channel), then combine across channels by and/or.

        Args:
            image: (H,W) or (H,W,C) uint8/float etc.
            q_range: (q_min, q_max)
            step: float step for q
            box_size: int
            target_mass: in (0,1]
            combine_mode: 'and' | 'or' across channels (if per-channel used)
            use_grayscale_measure: if True and image is 3ch, use BGR->GRAY for measure
            measure_mode:
                - "intensity_sum": m_i = sum of pixel values in box (requires nonnegative image for probabilistic meaning)

        Returns:
            mask_union: (H, W) bool
            masked_image: same shape as input image; outside mask set to 0
        """
        if image is None:
            raise ValueError("image is None")
        if box_size <= 0:
            raise ValueError("box_size must be positive")
        if step <= 0:
            raise ValueError("step must be positive")
        if not (0 < target_mass <= 1):
            raise ValueError("target_mass must be in (0, 1]")
        if combine_mode not in ("and", "or"):
            raise ValueError("combine_mode must be 'and' or 'or'")

        # Build channel list used for measure
        if image.ndim == 2:
            channels = [image]
        elif image.ndim == 3:
            if use_grayscale_measure:
                if image.shape[2] == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = np.mean(image, axis=2)
                channels = [gray]
            else:
                channels = [image[:, :, i] for i in range(image.shape[2])]
        else:
            raise ValueError("Unsupported image shape (only 2D/3D supported)")

        q_min, q_max = q_range
        q_list = np.arange(q_min, q_max + 1e-12, step).astype(np.float64)

        masks_all_channels = []

        for ch_img in channels:
            img = ch_img.astype(np.float64)
            h, w = img.shape

            h_crop = h - (h % box_size)
            w_crop = w - (w % box_size)

            if h_crop == 0 or w_crop == 0:
                masks_all_channels.append(np.zeros((h, w), dtype=bool))
                continue

            img_cropped = img[:h_crop, :w_crop]

            blocks = view_as_blocks(img_cropped, block_shape=(box_size, box_size))  # (nY,nX,bs,bs)

            if measure_mode == "intensity_sum":
                block_m = np.sum(blocks, axis=(2, 3))  # (nY,nX)
            else:
                raise ValueError(f"Unknown measure_mode: {measure_mode}")

            total = float(np.sum(block_m))

            # Require a valid positive total to interpret as probability measure
            if (not np.isfinite(total)) or total <= 0:
                masks_all_channels.append(np.zeros((h, w), dtype=bool))
                continue

            mu = block_m / total  # >=0 ideally, sum(mu)=1
            # If mu contains negatives (unexpected), deactivate them
            active = (mu > 0) & np.isfinite(mu)

            masks_q = []

            # Precompute log(mu) on active boxes
            log_mu = np.full_like(mu, -np.inf, dtype=np.float64)
            log_mu[active] = np.log(mu[active])

            for q in q_list:
                # weights on active boxes only: w_i = mu_i^q = exp(q*log(mu_i))
                exp_val = np.full_like(mu, -np.inf, dtype=np.float64)
                exp_val[active] = q * log_mu[active]

                # Stable normalization: subtract max over active
                if not np.any(active):
                    continue
                mmax = np.max(exp_val[active])
                if not np.isfinite(mmax):
                    continue

                #w = np.zeros_like(mu, dtype=np.float64)
                #w[active] = np.exp(exp_val[active] - mmax)
                
                weights = np.zeros_like(mu, dtype=np.float64)
                weights[active] = np.exp(exp_val[active] - mmax)
                
                s = float(np.sum(weights))
                if (not np.isfinite(s)) or s <= 0:
                    continue

                p = weights / s  # sums to 1 over active boxes

                flat = p.ravel()
                order = np.argsort(flat)[::-1]
                sorted_vals = flat[order]
                cumsum = np.cumsum(sorted_vals)

                k = int(np.searchsorted(cumsum, target_mass) + 1)
                k = min(k, flat.size)

                block_mask_flat = np.zeros_like(flat, dtype=bool)
                block_mask_flat[order[:k]] = True
                block_mask = block_mask_flat.reshape(p.shape)

                # Upsample block mask to pixel mask on cropped region
                mask_crop = np.repeat(np.repeat(block_mask, box_size, axis=0), box_size, axis=1)

                full_mask = np.zeros((h, w), dtype=bool)
                full_mask[:h_crop, :w_crop] = mask_crop
                masks_q.append(full_mask)

            if len(masks_q) == 0:
                masks_all_channels.append(np.zeros((h, w), dtype=bool))
            else:
                masks_all_channels.append(np.logical_or.reduce(masks_q))

        # Combine across channels
        if combine_mode == "and":
            mask_union = np.logical_and.reduce(masks_all_channels)
        else:
            mask_union = np.logical_or.reduce(masks_all_channels)

        # Apply mask
        masked_image = np.zeros_like(image)
        if image.ndim == 2:
            masked_image[mask_union] = image[mask_union]
        else:
            masked_image[mask_union, :] = image[mask_union, :]

        return mask_union, masked_image


# -----------------------------
# Demos
# -----------------------------
def demo_boxes_grayscale():
    """Simple demo for grayscale block split/merge/mask."""
    img = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(img, (128, 128), 80, 255, -1)

    block_size = (64, 64)
    boxes, raw_blocks = CFAImage.get_boxes_from_image(img, block_size, corp_type=-1)
    print("total boxes:", boxes.shape[0], "raw_blocks shape:", raw_blocks.shape)

    mask_pos = [(0, 0), (1, 1), (2, 2)]
    merged_image = CFAImage.get_image_from_boxes(raw_blocks)
    mask_image = CFAImage.get_mask_from_boxes(raw_blocks, mask_pos)
    image_with_mask = (merged_image * mask_image).astype(np.uint8)

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes[0, 0].imshow(img, cmap="gray"); axes[0, 0].set_title("Raw Image"); axes[0, 0].axis("off")
    axes[0, 1].imshow(merged_image, cmap="gray"); axes[0, 1].set_title("Restored Image"); axes[0, 1].axis("off")
    axes[1, 0].imshow(mask_image, cmap="gray"); axes[1, 0].set_title("Mask Image"); axes[1, 0].axis("off")
    axes[1, 1].imshow(image_with_mask, cmap="gray"); axes[1, 1].set_title("Image With Mask"); axes[1, 1].axis("off")
    plt.tight_layout()
    plt.show()


def demo_roi(file_name="../images/face.png"):
    """Demo for ROI extraction by q on a color image loaded by OpenCV (BGR)."""
    image = cv2.imread(file_name, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(file_name)

    q_range = (-5, 5)

    mask_union, masked_image = CFAImage.get_roi_by_q(
        image=image,
        q_range=q_range,
        step=1.0,
        box_size=16,
        target_mass=0.90,
        combine_mode="or",
        use_grayscale_measure=True,
    )

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask_union.astype(np.uint8) * 255, cmap="gray")
    plt.title(f"ROI Mask (q ∈ [{q_range[0]}, {q_range[1]}])")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    plt.title("Extracted Region")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    demo_boxes_grayscale()
    demo_roi("../images/face.png")  # 修改为你的路径


if __name__ == "__main__":
    main()

