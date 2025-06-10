import cv2
import argparse
from FreeAeonFractal.FA2Dimension import CFA2Dimension,CFA2DMFS

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
    else:
        raise ValueError("Invalid mode. Use 'fd' or 'mfs'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute fractal dimension or multifractal spectrum from an image.")
    parser.add_argument("--image", type=str, help="Path to the input image")
    parser.add_argument("--mode", choices=['fd', 'mfs'], default='mfs',
                        help="Choose 'fd' to compute fractal dimension or 'mfs' for multifractal spectrum (default: mfs)")

    args = parser.parse_args()

    main(args.image, args.mode)
