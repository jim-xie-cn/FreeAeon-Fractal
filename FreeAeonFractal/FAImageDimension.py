'''
Basic operations for 2D shapes
1. Calculation of various fractal dimensions
'''
import matplotlib.pyplot as plt
import numpy as np
import cv2,math,json,os,sys
from tqdm import tqdm
from scipy.stats import linregress
import pandas as pd
from FreeAeonFractal.FAImage import CFAImage
import seaborn as sns
import matplotlib.ticker as mticker
from scipy.interpolate import UnivariateSpline
'''
Calculation of fractal dimensions for 2D shapes
'''
class CFAImageDimension(object):
    '''
    image: input image (single channel)
    max_size: maximum box size for partitioning
    '''
    def __init__(self, image = None, max_size = None , max_scales = 100 , with_progress= True) :
        self.m_image = image
        if max_size == None:
            max_size = min(image.shape) // 1
        self.m_with_progress = with_progress
        scales = np.logspace(1, np.log2(max_size), num=max_scales, base=2, dtype=int)
        scales = np.unique(scales)
        self.m_scales = []
        for s in scales:
            if s > 0:
                self.m_scales.append(s)
    '''
    Linear regression fitting
    scale_list: list of box sizes
    box_count_list: list of box counts
    '''
    def get_fd(self,scale_list,box_count_list):
        s_list = np.array(scale_list)
        b_list = np.array(box_count_list)
        b_list = np.where(b_list == 0, np.finfo(float).eps, b_list)
 
        s_list = np.where(s_list == 0, np.finfo(float).eps, s_list)
        log_scales = -np.log(s_list)
        #log_scales = np.log(1 / np.array(s_list))
        log_counts = np.log(b_list)

        slope, intercept, r_value, p_value, std_err = linregress(log_scales, log_counts)
        ret = {}
        ret['fd'] = slope
        ret['scales'] = s_list.tolist()
        ret['counts'] = b_list.tolist()
        ret['log_scales'] = log_scales.tolist()
        ret['log_counts'] = log_counts.tolist()
        ret['intercept'] = intercept
        ret['r_value'] = r_value
        ret['p_value'] = p_value
        ret['std_err'] = std_err
        return ret

    '''Calculate fractal dimension using BC method
    corp_type: image cropping method (-1: crop, 0: no processing, 1: padding)
    '''
    def get_bc_fd(self,corp_type = -1):
        scale_list = []
        box_count_list = []
        if self.m_with_progress:
            for size in tqdm(self.m_scales ,desc="Calculating by BC"):
                block_size = (size,size)
                boxes,raw_blocks = CFAImage.get_boxes_from_image(self.m_image,block_size,corp_type=corp_type)
                axis = tuple(range(1, boxes.ndim))
                samples = np.sum(boxes, axis=axis)
                box_count = samples[samples>0].shape[0]
                scale_list.append(size)
                box_count_list.append(box_count)
        else:
            for size in self.m_scales:
                block_size = (size,size)
                boxes,raw_blocks = CFAImage.get_boxes_from_image(self.m_image,block_size,corp_type=corp_type)
                axis = tuple(range(1, boxes.ndim))
                samples = np.sum(boxes, axis=axis)
                box_count = samples[samples>0].shape[0]
                scale_list.append(size)
                box_count_list.append(box_count)

        return self.get_fd(scale_list,box_count_list)

    '''Calculate fractal dimension using DBC method
    corp_type: image cropping method (-1: crop, 0: no processing, 1: padding)
    '''
    def get_dbc_fd(self,corp_type = -1):
        scale_list = []
        box_count_list = []
        H = max(self.m_image.shape)
        #Gray_max = np.max(self.m_image)
        Gray_max = np.percentile(self.m_image, 99)
        if self.m_with_progress:
            for size in tqdm(self.m_scales,desc="Calculating by DBC"):
                block_size = (size,size)
                boxes,raw_blocks = CFAImage.get_boxes_from_image(self.m_image,block_size,corp_type=corp_type)
                box_count = 0
                # Calculate current box count
                for box in boxes:
                    I_min = np.min(box)
                    I_max = np.max(box)
                    # Calculate normalized height
                    #Z_min = (I_min / Gray_max) * H if I_min > 0 else 0
                    #Z_max = (I_max / Gray_max) * H if I_max > 0 else 0
                    Z_min = (I_min / Gray_max) * H
                    Z_max = (I_max / Gray_max) * H
                    #box_count += np.ceil((Z_max - Z_min) / size).astype(int)
                    delta_z = max(Z_max - Z_min, 0)
                    box_count += np.ceil((delta_z + 1e-6) / size).astype(int)

                scale_list.append(size)
                box_count_list.append(box_count)
        else:
            for size in self.m_scales: 
                block_size = (size,size)
                boxes,raw_blocks = CFAImage.get_boxes_from_image(self.m_image,block_size,corp_type=corp_type)
                box_count = 0
                # Calculate current box count
                for box in boxes:
                    I_min = np.min(box)
                    I_max = np.max(box)
                    # Calculate normalized height
                    #Z_min = (I_min / Gray_max) * H if I_min > 0 else 0
                    #Z_max = (I_max / Gray_max) * H if I_max > 0 else 0
                    Z_min = (I_min / Gray_max) * H
                    Z_max = (I_max / Gray_max) * H
                    #box_count += np.ceil((Z_max - Z_min) / size).astype(int)
                    delta_z = max(Z_max - Z_min, 0)
                    box_count += np.ceil((delta_z + 1e-6) / size).astype(int)
                scale_list.append(size)
                box_count_list.append(box_count)
        return self.get_fd(scale_list,box_count_list)

    '''Calculate fractal dimension using SDBC method
    corp_type: image cropping method (-1: crop, 0: no processing, 1: padding)
    '''
    def get_sdbc_fd(self, corp_type = -1):
        scale_list = []
        box_count_list = []
        H = max(self.m_image.shape)
        #Gray_max = np.max(self.m_image)
        Gray_max = np.percentile(self.m_image, 99)
        if self.m_with_progress:
            for size in tqdm(self.m_scales,desc="Calculating by SDBC"):
                block_size = (size,size)
                boxes,raw_blocks = CFAImage.get_boxes_from_image(self.m_image,block_size,corp_type=corp_type)
                box_count = 0
                # Calculate current box count
                for box in boxes:
                    I_min = np.min(box)
                    I_max = np.max(box)
                    # Calculate normalized height
                    #Z_min = (I_min / Gray_max) * H if I_min > 0 else 0
                    #Z_max = (I_max / Gray_max) * H if I_max > 0 else 0
                    Z_min = (I_min / Gray_max) * H
                    Z_max = (I_max / Gray_max) * H
                    #box_count += np.ceil(((Z_max-Z_min+1) * H ) / size).astype(int)
                    delta_z = max(Z_max - Z_min, 0)
                    box_count += np.ceil((delta_z + 1e-6) / size).astype(int)
                scale_list.append(size)
                box_count_list.append(box_count)
        else:
            for size in self.m_scales:
                block_size = (size,size)
                boxes,raw_blocks = CFAImage.get_boxes_from_image(self.m_image,block_size,corp_type=corp_type)
                box_count = 0
                # Calculate current box count
                for box in boxes:
                    I_min = np.min(box)
                    I_max = np.max(box)
                    # Calculate normalized height
                    #Z_min = (I_min / Gray_max) * H if I_min > 0 else 0
                    #Z_max = (I_max / Gray_max) * H if I_max > 0 else 0
                    Z_min = (I_min / Gray_max) * H
                    Z_max = (I_max / Gray_max) * H
                    #box_count += np.ceil(((Z_max-Z_min+1) * H ) / size).astype(int)
                    delta_z = max(Z_max - Z_min, 0)
                    box_count += np.ceil((delta_z + 1e-6) / size).astype(int)

                scale_list.append(size)
                box_count_list.append(box_count)

        return self.get_fd(scale_list, box_count_list)

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
            plt.title('%s: FD=%0.4f PV=%.4f' % (text,fd,result['p_value']),fontsize=7)
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
    raw_image = cv2.imread("../images/face.png", cv2.IMREAD_GRAYSCALE)
    bin_image = (raw_image >= 64).astype(int)     
    fd_bc = CFAImageDimension(bin_image).get_bc_fd(corp_type=-1)
    fd_dbc = CFAImageDimension(raw_image).get_dbc_fd(corp_type=-1)
    fd_sdbc = CFAImageDimension(raw_image).get_sdbc_fd(corp_type=-1)
    CFAImageDimension.plot(raw_image,bin_image,fd_bc,fd_dbc,fd_sdbc)

if __name__ == "__main__":
    main()
