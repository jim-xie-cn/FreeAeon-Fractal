import cv2
import argparse
from FreeAeonFractal.FA2Dimension import CFA2Dimension,CFA2DMFS
from FreeAeonFractal.FAImageFourier import CFAImageFourier

def main(image_path, mode):
    raw_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if raw_image is None:
        raise FileNotFoundError(f"Cannot load image at {image_path}")
    
    if mode == 'fd':
        bin_image = (raw_image >= 64).astype(int)
        fd_bc = CFA2Dimension(bin_image).get_bc_fd(corp_type=-1)
        fd_dbc = CFA2Dimension(raw_image).get_dbc_fd(corp_type=-1)
        fd_sdbc = CFA2Dimension(raw_image).get_sdbc_fd(corp_type=-1)
        CFA2Dimension.plot(raw_image, bin_image, fd_bc, fd_dbc, fd_sdbc)

    elif mode == 'mfs':
        MFS = CFA2DMFS(raw_image)
        df_mass, df_mfs = MFS.get_mfs()
        MFS.plot(df_mass, df_mfs)

    elif mode == 'fourier':
        # Create CFAImageFourier instance
        fourier = CFAImageFourier(raw_image)

        # Get display spectrum
        mag_disp, phase_disp = fourier.get_display_spectrum(alpha=1.5)

        # Reconstruct full image
        reconstructed = fourier.get_reconstruct()

        # Get region by maq or phase
        h, w = raw_image.shape[0], raw_image.shape[1]
        mag_box = (0,0,w//2,h//2)
        phase_box = (0,0,w,h)

        region_mag = fourier.extract_by_freq(box=mag_box)
        region_phase = fourier.extract_by_phase(box=phase_box)
        region_mag_phase = fourier.extract_by_freq_phase(mag_box,phase_box)
   
        # Show full result
        fourier.show(mag_disp, phase_disp, reconstructed, region_mag, region_phase,region_mag_phase)

    else:
        raise ValueError("Invalid mode. Use 'fd' or 'mfs' or 'fourier'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute fractal dimension or multifractal spectrum from an image.")
    parser.add_argument("--image", type=str,required=True, help="Path to the input image")
    parser.add_argument("--mode", choices=['fd', 'mfs', 'fourier'], default='mfs',
                        help="Choose 'fd' to compute fractal dimension, 'mfs' for multifractal spectrum or 'fourier' for Fourier analysis. (default: mfs)")

    args = parser.parse_args()

    main(args.image, args.mode)
