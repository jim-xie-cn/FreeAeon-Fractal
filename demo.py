import cv2
from FreeAeonFractal.FA2Dimension import CFA2Dimension,CFA2DMFS

def main():
    raw_image = cv2.imread("./images/face.png", cv2.IMREAD_GRAYSCALE)
    bin_image = (raw_image >= 64).astype(int)     
    fd_bc = CFA2Dimension(bin_image).get_bc_fd(corp_type=-1)
    fd_dbc = CFA2Dimension(raw_image).get_dbc_fd(corp_type=-1)
    fd_sdbc = CFA2Dimension(raw_image).get_sdbc_fd(corp_type=-1)
    CFA2Dimension.plot(raw_image,bin_image,fd_bc,fd_dbc,fd_sdbc)

    image = cv2.imread("./images/fractal.png", cv2.IMREAD_GRAYSCALE)
    MFS = CFA2DMFS(image)
    df_mass,df_mfs = MFS.get_mfs()
    MFS.plot(df_mass,df_mfs)


if __name__ == "__main__":
    main()


