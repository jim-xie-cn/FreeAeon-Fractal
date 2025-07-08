'''
Segmenting and Merging the Image
'''
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.ndimage
from skimage.util import view_as_blocks
from tqdm import tqdm
from scipy.stats import linregress
from FreeAeonFractal.FASample import CFASample

class CFAImage(object):
    """
    Crop the input data so that its shape is a multiple of the block size.

    Parameters
    ----------
    data : ndarray
        Input array (e.g., image or tensor).
    block_size : tuple of int
        Desired block size in each dimension.

    Returns
    -------
    cropped_data : ndarray
        Cropped array with dimensions being multiples of block_size.
    """
    @staticmethod
    def crop_data(data, block_size):
        new_shape = [data.shape[i] // block_size[i] * block_size[i] for i in range(len(data.shape))]
        slices = tuple(slice(0, new_shape[i]) for i in range(len(data.shape)))
        cropped_data = data[slices]
        return cropped_data

    """
    Pad the input data so that its shape becomes a multiple of the block size.

    Parameters
    ----------
    data : ndarray
        Input array.
    block_size : tuple of int
        Desired block size in each dimension.
    mode : str, optional
        Padding mode to use (default is 'constant').
    constant_values : int or float, optional
        Padding value to use if mode is 'constant' (default is 0).

    Returns
    -------
    padded_data : ndarray
        Padded array with dimensions being multiples of block_size.
    """
    @staticmethod
    def pad_data(data, block_size, mode='constant', constant_values=0):
        pad_width = []
        for i in range(len(data.shape)): 
            remainder = data.shape[i] % block_size[i]
            if remainder == 0:
                pad_width.append((0, 0))  
            else:
                pad_size = block_size[i] - remainder 
                pad_width.append((0, pad_size)) 
        padded_data = np.pad(data, pad_width, mode=mode, constant_values=constant_values)
        return padded_data
    
    """
    Divide the image into blocks of the specified size.

    Parameters
    ----------
    image : ndarray
        Input image or array.
    block_size : tuple of int
        Size of each block.
    corp_type : int, optional
        Handling strategy if image size is not a multiple of block_size:
        -1: crop the image,
            0: no change,
            1: pad the image.

    Returns
    -------
    blocks_reshaped : ndarray
        Flattened array of blocks with shape (N_blocks, *block_size).
    raw_blocks : ndarray
        Raw block-view of the image with shape corresponding to the block grid layout.
    """
    @staticmethod
    def get_boxes_from_image(image, block_size, corp_type=-1):
        if corp_type == -1:
            corp_data = CFAImage.crop_data(image, block_size)
        elif corp_type == 1:
            corp_data = CFAImage.pad_data(image, block_size)
        else:
            corp_data = image
        raw_blocks = view_as_blocks(corp_data, block_shape=block_size)
        num_blocks = np.prod(raw_blocks.shape[:len(block_size)])
        blocks_reshaped = raw_blocks.reshape(num_blocks, *block_size)
        return blocks_reshaped, raw_blocks
    
    """
    Merge a set of blocks back into a single image.

    Parameters
    ----------
    raw_blocks : ndarray
        Original block layout as returned by `get_boxes_from_image`.

    Returns
    -------
    merged_image : ndarray
        Reconstructed image from the block layout.
    """
    @staticmethod
    def get_image_from_boxes(raw_blocks):
        shape = raw_blocks.shape
        block_size = (shape[2], shape[3])
        num_blocks_y, num_blocks_x = shape[0], shape[1]
        H = num_blocks_y * raw_blocks.shape[2]
        W = num_blocks_x * raw_blocks.shape[3]
        merged_image = raw_blocks.transpose(0, 2, 1, 3).reshape(H, W)
        return merged_image
    
    """
    Generate a binary mask by selecting specific blocks.

    Parameters
    ----------
    raw_blocks : ndarray
        Original block layout as returned by `get_boxes_from_image`.
    mask_block_pos : list of tuple of int
        List of (y, x) positions indicating blocks to mask (set to 0).

    Returns
    -------
    mask_image : ndarray
        A binary mask image where selected blocks are 0 and others are 1.
    """
    @staticmethod
    def get_mask_from_boxes(raw_blocks, mask_block_pos):
        mask_block = []
        pos_h = 0
        for y_block in raw_blocks:
            tmp_block = []
            pos_w = 0
            for x_block in y_block:
                pos = (pos_h, pos_w) 
                if pos in map(tuple, mask_block_pos):
                    box = np.zeros(x_block.shape)
                else:
                    box = np.ones(x_block.shape)
                tmp_block.append(box)
                pos_w += 1
            mask_block.append(tmp_block)
            pos_h += 1
        
        mask_block = np.array(mask_block)
        return CFAImage.get_image_from_boxes(mask_block)
    
    """
    Extracts joint regions with high μ^q values over a range of q values.
    Supports both grayscale and RGB images.

    Args:
        image (ndarray): Input image (grayscale or RGB).
        q_range (tuple): Range of q values (q_min, q_max).
        step (float): Step size for iterating through q values.
        window_size (int): Smoothing window size.
        target_mass (float): Cumulative mass threshold to retain. 
                             Is it selecting the top 95 percentile of blocks based on the sorted mass values。
                             Less target_mass, Less boxes selected.

        combine_mode ('and'|'or'): Merge method for each channel.

    Returns:
        masked_image (ndarray): Image with only the significant regions (others set to 0).
        mask_union (ndarray): Final combined binary mask (2D boolean array).
    """
    @staticmethod
    def get_roi_by_q(image, q_range=(-5, 5), step=1, box_size=16, target_mass=0.95,combine_mode='and'):
        if image is None:
            raise ValueError("image is None")

        if image.ndim == 2:  # Grayscale
            channels = [image]
        elif image.ndim == 3 and image.shape[2] == 3:  # RGB
            channels = [image[:, :, i] for i in range(3)]
        else:
            raise ValueError("Unsupported image shape")

        q_min, q_max = q_range
        q_list = np.arange(q_min, q_max + step, step)
        eps = 1e-12

        masks_all_channels = []

        for ch_img in channels:
            img = ch_img.astype(np.float64)  
            h, w = img.shape
            h_crop = h - (h % box_size)
            w_crop = w - (w % box_size)
            img_cropped = img[:h_crop, :w_crop]

            blocks = view_as_blocks(img_cropped, block_shape=(box_size, box_size))
            block_sum = np.sum(blocks, axis=(2, 3))
            block_sum = np.maximum(block_sum, eps)
            log_block_sum = np.log(block_sum)

            masks = []

            for q_val in q_list:
                exp_val = q_val * log_block_sum
                max_exp_val = np.max(exp_val)
                exp_val_stable = exp_val - max_exp_val  # handle extreme large number

                mass_q = np.exp(exp_val_stable)
                mass_q = np.nan_to_num(mass_q, nan=0.0, posinf=np.finfo(np.float64).max, neginf=0.0)

                sum_mass_q = np.sum(mass_q)
                if sum_mass_q <= 0 or np.isnan(sum_mass_q):
                    continue

                mass_q_norm = mass_q / sum_mass_q

                flat = mass_q_norm.flatten()
                sorted_indices = np.argsort(flat)[::-1]
                sorted_vals = flat[sorted_indices]
                cumsum = np.cumsum(sorted_vals)
                cutoff_idx = np.searchsorted(cumsum, target_mass)

                block_mask_flat = np.zeros_like(flat, dtype=bool)
                block_mask_flat[sorted_indices[:cutoff_idx]] = True
                block_mask = block_mask_flat.reshape(mass_q.shape)

                mask_img = np.kron(block_mask, np.ones((box_size, box_size), dtype=bool))

                full_mask = np.zeros_like(img, dtype=bool)
                full_mask[:h_crop, :w_crop] = mask_img
                masks.append(full_mask)

            mask_channel = np.logical_or.reduce(masks)
            masks_all_channels.append(mask_channel)

        if combine_mode == 'and':
            mask_union = np.logical_and.reduce(masks_all_channels)
        elif combine_mode == 'or':
            mask_union = np.logical_or.reduce(masks_all_channels)
        else:
            raise ValueError("combine_mode must be 'and' or 'or'")

        masked_image = np.zeros_like(image)
        if image.ndim == 2:
            masked_image[mask_union] = image[mask_union]
        else:
            for i in range(3):
                ch = image[:, :, i]
                ch_masked = np.zeros_like(ch)
                ch_masked[mask_union] = ch[mask_union]
                masked_image[:, :, i] = ch_masked

        return mask_union, masked_image.astype(np.uint8)

def demo_boxes():
    points = CFASample.get_Sierpinski_Triangle(iterations = 1024)
    image = CFASample.get_image_from_points(points)

    block_size = (64, 64)
    boxes,raw_blocks = CFAImage.get_boxes_from_image(image, block_size, -1)
    print("total boxes:",boxes.shape[0])

    mask_pos = [(0, 0), (10, 10), (24, 24)]
    merged_image = CFAImage.get_image_from_boxes(raw_blocks)
    mask_image = CFAImage.get_mask_from_boxes(raw_blocks, mask_pos)
    image_with_mask = merged_image * mask_image

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title("Raw Image")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(merged_image, cmap='gray')
    axes[0, 1].set_title("Restored Image")
    axes[0, 1].axis('off')

    axes[1, 0].imshow(mask_image, cmap='gray')
    axes[1, 0].set_title("Masked Image")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(image_with_mask, cmap='gray')
    axes[1, 1].set_title("Image With Mask")
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

def demo_roi():
    file_name = './images/face.png'
    image  = cv2.imread(file_name, cv2.IMREAD_COLOR)
    b,g,r = cv2.split(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    q_range = (2,5)
    mask_union,masked_image = CFAImage.get_roi_by_q(image=image,
                                                    q_range=q_range,
                                                    step=1.0,
                                                    target_mass=0.90,
                                                    combine_mode='or')

    plt.figure(figsize=(12, 4))
        
    plt.subplot(1, 3, 1)
    if image.ndim == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask_union.astype(np.uint8)*255, cmap='gray')
    plt.title(f"Union Mask (q ∈ [{q_range[0]}, {q_range[1]}])")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    if masked_image.ndim == 2:
        plt.imshow(masked_image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(masked_image.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.title("Extracted Region")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def main():
    demo_boxes()
    demo_roi()

if __name__ == "__main__":
    main()
