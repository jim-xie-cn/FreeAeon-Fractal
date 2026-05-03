import cv2
import argparse
import numpy as np
import time
from FreeAeonFractal.FAImageFourier import CFAImageFourier
from FreeAeonFractal.FAImage import CFAImage
from FreeAeonFractal.FASeriesMFS import CFASeriesMFS

#CPU version
from FreeAeonFractal.FAImageLAC import CFAImageLAC
from FreeAeonFractal.FAImageFD import CFAImageFD
from FreeAeonFractal.FAImageMFS import CFAImageMFS

#GPU version
#from FreeAeonFractal.FAImageLACGPU import CFAImageLACGPU as CFAImageLAC
#from FreeAeonFractal.FAImageFDGPU import CFAImageFDGPU as CFAImageFD
#from FreeAeonFractal.FAImageMFSGPU import CFAImageMFSGPU as CFAImageMFS

def demo_series_mfs():
    x = np.cumsum(np.random.randn(5000))
    q = np.linspace(-5, 5, 21)
    mfs = CFASeriesMFS(x)
    df_mfs = mfs.get_mfs()
    mfs.plot(df_mfs)
    print(df_mfs)

def demo_fd(image_path):
    rgb_image = cv2.imread(image_path)
    if rgb_image is None:
        raise FileNotFoundError(f"Cannot load image")
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    bin_image,threshold = CFAImage.otsu_binarize(gray_image)

    max_scales = 32
    #  ---- single ----
    t0 = time.time()
    fd_bc = CFAImageFD(bin_image,max_scales=max_scales).get_bc_fd(corp_type=-1)
    fd_dbc = CFAImageFD(gray_image,max_scales=max_scales).get_dbc_fd(corp_type=-1)
    fd_sdbc = CFAImageFD(gray_image,max_scales=max_scales).get_sdbc_fd(corp_type=-1)
    print(f"Single (1) img: {time.time()-t0:.3f}s")
    print("  BC:",fd_bc['fd'])
    print("  DBC:",fd_dbc['fd'])
    print("  SDBC:",fd_sdbc['fd'])

    CFAImageFD.plot(gray_image, bin_image, fd_bc, fd_dbc, fd_sdbc)
    # ---- batch ----
    bin_imgs = [bin_image] * 100
    gray_imgs = [gray_image] * 100
    t0 = time.time()
    bc_list = CFAImageFD.get_batch_bc(bin_imgs, max_scales=max_scales,with_progress=False)
    dbc_list = CFAImageFD.get_batch_dbc(gray_imgs, max_scales=max_scales,with_progress=False)
    sdbc_list = CFAImageFD.get_batch_sdbc(gray_imgs, max_scales=max_scales,with_progress=False)
    print(f"Batch (100 imgs): {time.time()-t0:.3f}s")
    print(f"  batch BC FD[99]   = {bc_list[99]['fd']:.4f}")
    print(f"  batch DBC FD[99]  = {dbc_list[99]['fd']:.4f}")
    print(f"  batch SDBC FD[99] = {sdbc_list[99]['fd']:.4f}")
    CFAImageFD.plot(gray_image, bin_image, bc_list[99], dbc_list[99], sdbc_list[99])
 
def demo_mfs(image_path):
    rgb_image = cv2.imread(image_path)
    if rgb_image is None:
        raise FileNotFoundError(f"Cannot load image")
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    
    q_list = np.linspace(-5, 5, 51)
    # --- single ----
    t0 = time.time()
    MFS = CFAImageMFS(gray_image,q_list = q_list )
    df_mass, df_fit, df_spec = MFS.get_mfs()
    print(f"Single MFS (1) imgs:{time.time()-t0:.3f}s")
    print(df_fit.head())
    MFS.plot(df_mass,df_fit,df_spec)
    
    # ---- batch ----
    t0 = time.time()
    imgs = [gray_image] * 20
    batch_results = CFAImageMFS.get_batch_mfs( imgs, 
            with_progress=False, q_list=q_list, corp_type=-1,
            bg_reverse=False, bg_threshold=0.01, bg_otsu=False, max_scales=80,
            min_points=6, use_middle_scales=False, if_auto_line_fit=False,
            fit_scale_frac=(0.3, 0.7), auto_fit_min_len_ratio=0.6,
                                                    cap_d0_at_2=False)
    df_mass1, df_fit1, df_spec1 = batch_results[0]
    print(f"Batch MFS (20) imgs:{time.time()-t0:.3f}s" )
    print(df_fit1.head())
    MFS.plot(df_mass1,df_fit1,df_spec1)

def demo_alpha(image_path):
    rgb_image = cv2.imread(image_path)
    if rgb_image is None:
        raise FileNotFoundError(f"Cannot load image")
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    q_list = np.linspace(-5, 5, 51)
    # --- single ----
    t0 = time.time()
    MFS = CFAImageMFS(gray_image,q_list = q_list )
    alpha_map, info = MFS.compute_alpha_map(scales=[2, 4, 8, 16, 32])
    print(f"Single alpha_map (1) imgs:{time.time()-t0:.3f}s")
    print("  alpha map:",alpha_map)
    print("  scale info",info)
    CFAImageMFS.plot_alpha_map(alpha_map)

    # ---- batch ----
    t0 = time.time()
    imgs = [gray_image] * 20
    t0 = time.time()
    batch_alpha_map = CFAImageMFS.compute_alpha_map_batch(imgs,with_progress=False, scales=[2, 4, 8, 16, 32])
    alpha_maps = batch_alpha_map[0]
    infos = batch_alpha_map[1]
    print(f"Batch alpha_map (20) imgs:{time.time()-t0:.3f}s" )
    print("  alpha map:",alpha_maps[0])
    print("  scale info",infos)
    CFAImageMFS.plot_alpha_map(alpha_maps[0])

def demo_lacunarity(image_path):
    rgb_image = cv2.imread(image_path)
    if rgb_image is None:
        raise FileNotFoundError(f"Cannot load image")
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)   
    lacunarity = CFAImageLAC(gray_image,max_scales=256, with_progress=True) 
    # ---- Single ----
    t0 = time.time()
    lac_gray = lacunarity.get_lacunarity(corp_type=-1, use_binary_mass=False, include_zero=True)
    fit_gray = lacunarity.fit_lacunarity(lac_gray)
    print(f"Single lacunarity (1) imgs:{time.time()-t0:.3f}s" )
    print("  Gray lacunarity:", lac_gray["lacunarity"])
    print("  Fit slope:", fit_gray["slope"],"Fit intercept",fit_gray["intercept"],  "R:", fit_gray["r_value"],"P:",fit_gray["p_value"])
    lacunarity.plot(lac_gray,fit_gray)

    # ---- batch ----
    t0 = time.time()
    imgs = [gray_image] * 100
    batchs = CFAImageLAC.get_batch_lacunarity(
        imgs, scales_mode="powers", partition_mode="gliding",
        use_binary_mass=False, with_progress=False)
    fits = CFAImageLAC.fit_batch_lacunarity(batchs)
    print(f"Batch lacunarity (100) imgs:{time.time()-t0:.3f}s" )
    print("  Gray lacunarity:", batchs[99]["lacunarity"])
    print("  Fit slope:", fits[99]["slope"],"Fit intercept",fits[99]["intercept"],  "R:", fits[99]["r_value"],"P:",fits[99]["p_value"])

def demo_fourier(image_path):  
    rgb_image = cv2.imread(image_path)
    if rgb_image is None:
        raise FileNotFoundError(f"Cannot load image")
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    #fourier for gray images
    #gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    #fourier = CFAImageFourier(gray_image)

    #fourier for RGB images
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
    print(masked_reconstructed)

def main(image_path, mode):
    if mode == 'fd':
        demo_fd(image_path)
    elif mode == 'mfs':
        demo_mfs(image_path)
    elif mode == 'alpha':
        demo_alpha(image_path)
    elif mode == 'fourier':
        demo_fourier(image_path)
    elif mode == 'lacunarity':
        demo_lacunarity(image_path)
    elif mode == 'series':
        demo_series_mfs()
    else:
        raise ValueError("Invalid mode. Use 'fd' or 'mfs' or 'lacunarity' or 'fourier'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute fractal dimension or multifractal spectrum from an image.")
    parser.add_argument("--image", type=str, help="Path to the input image",default='./images/fractal.png')
    parser.add_argument("--mode", choices=['fd','mfs','alpha','lacunarity', 'fourier','series'], default='mfs',
                        help="Choose 'fd' to compute fractal dimension, 'mfs' for multifractal spectrum,'alpha' for local alpha map or 'fourier' for Fourier analysis. (default: mfs)")

    args = parser.parse_args()

    # Conditional check: image is required unless mode is 'series'
    if args.mode != 'series' and args.image is None:
        parser.error("--image is required when mode is not 'series'.")

    main(args.image, args.mode)
