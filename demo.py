import cv2
import argparse
import numpy as np
from FreeAeonFractal.FAImageFourier import CFAImageFourier
#CPU version
from FreeAeonFractal.FAImageDimension import CFAImageDimension
from FreeAeonFractal.FA2DMFS import CFA2DMFS
#GPU version
#from FreeAeonFractal.FAImageDimension import CFAImageDimensionGPU as CFAImageDimension
#from FreeAeonFractal.FA2DMFSGPU import CFA2DMFSGPU as CFA2DMFS

from FreeAeonFractal.FA1DMFS import CFA1DMFS

def demo_1d_mfs():
    x = np.cumsum(np.random.randn(5000))
    q = np.linspace(-5, 5, 21)
    mfs = CFA1DMFS(x)
    df_mfs = mfs.get_mfs()
    mfs.plot(df_mfs)

def demo_2d_fd(image_path):
    rgb_image = cv2.imread(image_path)
    if rgb_image is None:
        raise FileNotFoundError(f"Cannot load image")
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    bin_image = (gray_image >= 64).astype(int)
    fd_bc = CFAImageDimension(bin_image).get_bc_fd(corp_type=-1)
    fd_dbc = CFAImageDimension(gray_image).get_dbc_fd(corp_type=-1)
    fd_sdbc = CFAImageDimension(gray_image).get_sdbc_fd(corp_type=-1)
    CFAImageDimension.plot(gray_image, bin_image, fd_bc, fd_dbc, fd_sdbc)

def demo_2d_mfs(image_path):
    rgb_image = cv2.imread(image_path)
    if rgb_image is None:
        raise FileNotFoundError(f"Cannot load image")
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    MFS = CFA2DMFS(gray_image,q_list = np.linspace(0, 5, 26) )
    df_mass, df_fit, df_spec = MFS.get_mfs()
    MFS.plot(df_mass,df_fit,df_spec)

def demo_fourier(image_path):  
    rgb_image = cv2.imread(image_path)
    if rgb_image is None:
        raise FileNotFoundError(f"Cannot load image")
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    #fourier for grayscale
    #gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    #demo_fourier(gray_image)

    #fourier for RGB

    # Create CFAImageFourier instance
    fourier = CFAImageFourier(rgb_image)

    # Get raw spectrum
    raw_mag, raw_phase = fourier.get_raw_spectrum()

    # Get display spectrum
    raw_mag_disp, raw_phase_disp = fourier.get_display_spectrum(alpha=1.5)

    # Fake mask (reserve odd frequencies))
    h, w = raw_mag[0].shape
    Y, X = np.ogrid[:h, :w]
    mask = ((X % 2 == 1) & (Y % 2 == 1)).astype(np.uint8)

    # Get masked display spectrum
    customized_mag_list = raw_mag * mask
    customized_phase_list = raw_phase * mask
    customized_mag_disp, customized_phase_disp = fourier.get_display_spectrum(alpha=1.5,
                                                                              magnitude = customized_mag_list, 
                                                                              phase = customized_phase_list)
    # Reconstruct full image
    full_reconstructed = fourier.get_reconstruct()

    #Reconstructet image by frequency mask 
    masked_reconstructed = fourier.extract_by_freq_mask(mask)

    # Show full result
    fourier.plot(raw_mag_disp, 
                 raw_phase_disp, 
                 customized_mag_disp,
                 customized_phase_disp,
                 full_reconstructed, 
                 masked_reconstructed)
    
def main(image_path, mode):
    if mode == 'fd':
        demo_2d_fd(image_path)
    elif mode == 'mfs':
        demo_2d_mfs(image_path)
    elif mode == 'fourier':
        demo_fourier(image_path)
    elif mode == 'series':
        demo_1d_mfs()
    else:
        raise ValueError("Invalid mode. Use 'fd' or 'mfs' or 'fourier'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute fractal dimension or multifractal spectrum from an image.")
    parser.add_argument("--image", type=str, help="Path to the input image")
    parser.add_argument("--mode", choices=['fd', 'mfs', 'fourier','series'], default='mfs',
                        help="Choose 'fd' to compute fractal dimension, 'mfs' for multifractal spectrum or 'fourier' for Fourier analysis. (default: mfs)")

    args = parser.parse_args()

    # Conditional check: image is required unless mode is 'series'
    if args.mode != 'series' and args.image is None:
        parser.error("--image is required when mode is not 'series'.")

    main(args.image, args.mode)
