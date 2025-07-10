import cv2
import argparse
import numpy as np
from FreeAeonFractal.FA2Dimension import CFA2Dimension,CFA2DMFS
from FreeAeonFractal.FAImageFourier import CFAImageFourier

def demo_fd(image):
    bin_image = (image >= 64).astype(int)
    fd_bc = CFA2Dimension(bin_image).get_bc_fd(corp_type=-1)
    fd_dbc = CFA2Dimension(image).get_dbc_fd(corp_type=-1)
    fd_sdbc = CFA2Dimension(image).get_sdbc_fd(corp_type=-1)
    CFA2Dimension.plot(image, bin_image, fd_bc, fd_dbc, fd_sdbc)

def demo_mfs(image):
    MFS = CFA2DMFS(image)
    df_mass, df_mfs = MFS.get_mfs()
    MFS.plot(df_mass, df_mfs) 

def demo_fourier(image):
    # Create CFAImageFourier instance
    fourier = CFAImageFourier(image)

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
    rgb_image = cv2.imread(image_path)
    if rgb_image is None:
        raise FileNotFoundError(f"Cannot load image")
    
    if mode == 'fd':
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        demo_fd(gray_image)
    elif mode == 'mfs':
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        demo_mfs(gray_image)
    elif mode == 'fourier':
        #fourier for grayscale
        #gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        #demo_fourier(gray_image)

        #fourier for RGB
        demo_fourier(rgb_image)
    else:
        raise ValueError("Invalid mode. Use 'fd' or 'mfs' or 'fourier'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute fractal dimension or multifractal spectrum from an image.")
    parser.add_argument("--image", type=str,required=True, help="Path to the input image")
    parser.add_argument("--mode", choices=['fd', 'mfs', 'fourier'], default='mfs',
                        help="Choose 'fd' to compute fractal dimension, 'mfs' for multifractal spectrum or 'fourier' for Fourier analysis. (default: mfs)")

    args = parser.parse_args()

    main(args.image, args.mode)
