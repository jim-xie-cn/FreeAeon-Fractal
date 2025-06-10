'''
Segmenting and Merging the Image
'''
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import view_as_blocks
from tqdm import tqdm
from scipy.stats import linregress
from .FASample import CFASample

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

def main():
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

if __name__ == "__main__":
    main()

plt.show()
