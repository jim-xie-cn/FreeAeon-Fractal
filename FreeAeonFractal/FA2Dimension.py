'''
Basic operations for 2D shapes
1. Calculation of various fractal dimensions
2. Calculation of multifractal spectra
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
class CFA2Dimension(object):
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

'''
Calculation of multifractal spectrum for 2D shapes
'''
class CFA2DMFS:
    '''
    image: input image (single channel)
    q_list: range of q values
    corp_type: image cropping method (-1: crop, 0: no processing, 1: padding)
    '''
    def __init__(self, image, corp_type = -1, q_list=np.linspace(-5, 5, 51) , with_progress= True ):
        self.m_image = image / np.max(image)  # normalize image
        self.m_corp_type = corp_type  # image cropping mode: -1 crop, 0 no processing, 1 padding
        self.m_q_list = []
        # avoid q == 1 case (ignored)
        for q in q_list:
           if q == 1:
               self.m_q_list.append( q )
               #self.m_q_list.append( q - np.finfo(float).eps )
           else:
               self.m_q_list.append( q )
        self.m_with_progress = with_progress

    '''Get generalized mass distribution
    max_size: max box size for partitioning
    Returns: generalized mass distribution
    '''
    def get_mass(self, max_size=None, max_scales=200):
        all_data = []

        if max_size is None:
            max_size = min(self.m_image.shape)  
        if max_size < 4:
            raise ValueError("max_size too small: must be >= 4")

        # get box size ( ε ) list
        scales = np.logspace(1, np.log2(max_size), num=max_scales, base=2, dtype=int)
        scales = np.unique(scales)

        q_list = self.m_q_list
        image = self.m_image.astype(np.float64)  # for large numbers
        progress_iter = tqdm(scales, desc="Calculating mass") if self.m_with_progress else scales

        for size in progress_iter:
            if size < 4:
                continue
            block_size = (size, size)
            boxes, raw_blocks = CFAImage.get_boxes_from_image(image, block_size, corp_type=self.m_corp_type)
            no_zero_box = [box for box in boxes if np.count_nonzero(box) > 0]
            if len(no_zero_box) == 0:
                continue

            # Step 1: Calculate boxes mass
            mass_distribution = np.array([np.sum(box) for box in boxes], dtype=np.float64)
            mass_distribution = np.where(mass_distribution < 0, 0, mass_distribution)

            total_mass = np.sum(mass_distribution)
            if total_mass <= 0 or np.isnan(total_mass) or np.isinf(total_mass):
                continue

            mass_distribution /= total_mass
            mass_distribution = np.clip(mass_distribution, 1e-12, 1.0) # Prevent zero or extreme values

            # Step 2: Calculate M(q, ε)
            for q in q_list:
                tmp = {'scale': size, 'q': q, 'boxes': raw_blocks}

                if np.all(mass_distribution == 0):
                    tmp['mass'] = 0
                    all_data.append(tmp)
                    continue

                if q == 0:
                    tmp['mass'] = np.count_nonzero(mass_distribution)
                else:
                    # Numerically stable computation: mass^q = exp(q * log(mass))
                    log_mass = np.log(mass_distribution)
                    q_log_mass = q * log_mass

                    # Limit extreme values to prevent exp overflow
                    q_log_mass = np.clip(q_log_mass, a_min=-700, a_max=700)
                    mass_q = np.sum(np.exp(q_log_mass))

                    if np.isnan(mass_q) or np.isinf(mass_q):
                        mass_q = 0
                    tmp['mass'] = mass_q

                all_data.append(tmp)

        df = pd.DataFrame(all_data)
        return df.sort_values(by=['q', 'scale']).reset_index(drop=True)
    
    '''Calculate scaling exponent tau
    df_mass: generalized mass distribution dataframe
    Returns: q list and scaling exponent (tau) list
    '''
    def get_tau_q(self,df_mass):
        all_data = []
        if self.m_with_progress:
            for q, df_q in tqdm(df_mass.groupby("q"),desc="Calculating τ(q)"):
                tmp = {}
                if q == 1:
                    #mass = np.array(df_q['mass'].tolist())
                    #mass = mass[mass>0]
                    #tau = -np.sum(mass * np.log(mass))
                    #tmp['q'] = q
                    #tmp['t(q)'] = tau
                    entropy_list = []
                    log_scales = []
                    for scale, df_scale in df_q.groupby('scale'):
                        mass = np.array(df_scale['mass'])
                        mass = mass[mass > 0]
                        entropy = -np.sum(mass * np.log(mass))
                        entropy_list.append(entropy)
                        log_scales.append(np.log(scale))
                    if len(log_scales) > 5:
                        slope, intercept, r_value, p_value, std_err = linregress(log_scales, entropy_list)
                        tau = slope
                        tmp['q'] = q
                        tmp['t(q)'] = slope
                        tmp['intercept'] = intercept
                        tmp['r_value'] = r_value
                        tmp['p_value'] = p_value
                        tmp['std_err'] = std_err
                    else:
                        tau = np.nan
                        tmp['q'] = q
                        tmp['t(q)'] = tau
                else:
                    log_scales = np.log(df_q['scale'])
                    log_mass = np.log(df_q['mass'])
                    log_mass = np.where(log_mass == -np.inf, np.nan, log_mass)
                    valid_mask = np.isfinite(log_mass)
                    log_mass_valid = log_mass[valid_mask]
                    log_scales_valid = log_scales[valid_mask]
                    if len(log_scales_valid) > 5:
                        slope, intercept, r_value, p_value, std_err = linregress(log_scales_valid, log_mass_valid)
                        #slope, _ = np.polyfit(log_scales_valid, log_mass_valid, 1)
                        tmp['q'] = q
                        tmp['t(q)'] = slope
                        tmp['intercept'] = intercept
                        tmp['r_value'] = r_value
                        tmp['p_value'] = p_value
                        tmp['std_err'] = std_err
                    else:
                        tmp['q'] = q
                        tmp['t(q)'] = np.nan
                all_data.append(tmp)
        else:
            for q, df_q in df_mass.groupby("q"):
                tmp = {}
                if q == 1:
                    mass = np.array(df_q['mass'].tolist())
                    mass = mass[mass>0]
                    tau = -np.sum(mass * np.log(mass))
                    tmp['q'] = q
                    tmp['t(q)'] = tau
                else:
                    log_scales = np.log(df_q['scale'])
                    log_mass = np.log(df_q['mass'])
                    log_mass = np.where(log_mass == -np.inf, np.nan, log_mass)
                    valid_mask = np.isfinite(log_mass)
                    log_mass_valid = log_mass[valid_mask]
                    log_scales_valid = log_scales[valid_mask]
                    if len(log_scales_valid) > 5:
                        slope, intercept, r_value, p_value, std_err = linregress(log_scales_valid, log_mass_valid)
                        #slope, _ = np.polyfit(log_scales_valid, log_mass_valid, 1)
                        tmp['q'] = q
                        tmp['t(q)'] = slope
                        tmp['intercept'] = intercept
                        tmp['r_value'] = r_value
                        tmp['p_value'] = p_value
                        tmp['std_err'] = std_err
                    else:
                        tmp['q'] = q
                        tmp['t(q)'] = np.nan
                all_data.append(tmp)
        return pd.DataFrame(all_data)

    '''Calculate the generalized fractal dimension
    item: a single record in tau (value corresponding to a certain q)
    Returns: generalized fractal dimension (D)
    '''
    def get_generalized_dimension(self,df_tau):
        df_tau = self.get_alpha_f_alpha(df_tau)
        def calc_dimension(item):
            if item['q'] == 1:
                return item['a(q)']  # 用 alpha(1) 作为 d(1)
            else:
                return item['t(q)'] / (item['q'] - 1)
        df_tau['d(q)'] = df_tau.apply(calc_dimension, axis=1)
        return df_tau
        #def calc_dimension(item):
        #    if item['q'] == 1:
        #        return item['t(q)']  # use tau(q) when q is 1
        #    else:
        #        return item['t(q)'] / (item['q'] - 1)
        #df_tau['d(q)'] = df_tau.progress_apply(calc_dimension, axis=1)
        #df_tau['d(q)'] = df_tau.apply(calc_dimension, axis=1)
        #return df_tau

    '''Calculate the local singularity exponent and singularity spectrum
    df_tau: scaling data DataFrame df_tau
    Returns: list of local singularity exponents (alpha) and singularity spectrum (f(alpha))
    '''
    def get_alpha_f_alpha(self,df_tau):
        #alpha_list = np.gradient(df_tau['t(q)'], df_tau['q'])
        #f_alpha_list = df_tau['q'] * alpha_list - df_tau['t(q)']
        q_vals = df_tau['q'].values
        tq_vals = df_tau['t(q)'].values
        spl = UnivariateSpline(q_vals, tq_vals, k=3, s=0)
        alpha_list = spl.derivative()(q_vals)
        f_alpha_list = q_vals * alpha_list - tq_vals

        df_tau['a(q)'] = alpha_list
        df_tau['f(a)'] = f_alpha_list
        return df_tau
    
    '''Calculate multifractal spectrum from tau(q)
    df_tau: dataframe of tau(q)
    Returns: dataframe with q, alpha, f(alpha)
    '''
    def get_mfs(self, max_size = None, max_scales = 100 ):
        df_mass = self.get_mass(max_size = max_size, max_scales = max_scales )
        df_mfs = self.get_tau_q(df_mass)
        df_mfs = self.get_generalized_dimension(df_mfs)
        df_mfs = self.get_alpha_f_alpha(df_mfs)
        return df_mass,df_mfs
    
    '''Plot multifractal spectrum'''
    def plot(self, df_mass,df_mfs):
        fig, axs = plt.subplots(2, 3, figsize=(16, 8))
        
        #  mass vs scale vs. q
        vmin = np.percentile(df_mass['mass'], 25)   #  5% percentage
        vmax = np.percentile(df_mass['mass'], 75)  #  95% percentage
        df_pivot = df_mass.pivot(index='scale', columns='q', values='mass')
        sns.heatmap(df_pivot,ax=axs[0, 0], cmap='coolwarm',vmin=vmin, vmax=vmax, annot=False, cbar=True)
        axs[0, 0].set(xlabel=r'$scale$', ylabel=r'$q$', title=r'mass vs. scale vs. q')
        axs[0, 0].xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        axs[0, 0].set_xticklabels(axs[0, 0].get_xticklabels(), rotation=60)
        axs[0, 0].grid(True)

        #  tau(q) vs. q
        df_tmp = df_mfs.groupby("q").mean().reset_index().sort_values(by = "q").reset_index()
        sns.lineplot(data=df_tmp, x='q', y='t(q)', ax=axs[0, 1])
        axs[0, 1].set(xlabel='$q$', ylabel=r'$\tau(q)$', title=r'$\tau(q)$ vs. $q$')
        axs[0, 1].grid(True)

        #  D(q) vs. q
        df_tmp = df_mfs.groupby("q").mean().reset_index().sort_values(by = "q").reset_index()
        sns.lineplot(data=df_tmp, x='q', y='d(q)', ax=axs[0, 2])
        axs[0, 2].set(xlabel='$q$', ylabel=r'$D(q)$', title=r'$D(q)$ vs. $q$')
        axs[0, 2].grid(True)

        #  α(q) vs. q
        df_tmp = df_mfs.groupby("q").mean().reset_index().sort_values(by = "q").reset_index()
        sns.lineplot(data=df_tmp, x='q', y='a(q)', ax=axs[1, 0])
        axs[1, 0].set(xlabel='$q$', ylabel=r'$\alpha$', title=r'$\alpha(q)$ vs. $q$')
        axs[1, 0].grid(True)

        #  f(α) vs. α
        df_tmp = df_mfs.groupby("a(q)").mean().reset_index().sort_values(by = "a(q)").reset_index()
        sns.lineplot(data=df_tmp, x='a(q)', y='f(a)', ax=axs[1, 1])
        axs[1, 1].set(xlabel=r'$\alpha$', ylabel=r'f$(\alpha)$', title=r'$f(\alpha)$ vs. $\alpha$')
        axs[1, 1].grid(True)

        #  f(α) vs. d
        df_tmp = df_mfs.groupby("q").mean().reset_index().sort_values(by = "q").reset_index()
        df_tmp['tmp'] = (df_tmp['t(q)'] - df_tmp['t(q)'].min()) / (df_tmp['t(q)'].std())
        sns.lineplot(data=df_tmp, x='q', y='tmp', ax=axs[1, 2],label=r'$t(q)$')

        df_tmp['tmp'] = (df_tmp['d(q)'] - df_tmp['d(q)'].min()) / (df_tmp['d(q)'].std())
        sns.lineplot(data=df_tmp, x='q', y='tmp', ax=axs[1, 2],label=r'$d(q)$')

        df_tmp['tmp'] = (df_tmp['a(q)'] - df_tmp['a(q)'].min()) / (df_tmp['a(q)'].std())
        sns.lineplot(data=df_tmp, x='q', y='tmp', ax=axs[1, 2],label=r'$a(q)$')

        df_tmp['tmp'] = (df_tmp['f(a)'] - df_tmp['f(a)'].min()) / (df_tmp['f(a)'].std())
        sns.lineplot(data=df_tmp, x='q', y='tmp', ax=axs[1, 2],label=r'$f(a)$')

        axs[1, 2].set(xlabel=r'$q$', ylabel=r'$mix$', title=r'$overview$ vs. $q$')
        axs[1, 2].grid(True)
        
        plt.tight_layout()
        plt.show()

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
