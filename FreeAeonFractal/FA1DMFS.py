import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math,json,sys,os
from tqdm import tqdm
tqdm.pandas()
from MFDFA import MFDFA

def recommended_lag(x_len, order=2, num_scales=40, s_min=None, s_max_ratio=0.25):
    """
    Generate a dense set of scale windows for medium to short sequences,
    designed for high-precision multifractal analysis.

    Parameters:
        x_len        : Length of the input sequence
        order        : Polynomial fitting order for DFA (default: 2)
        num_scales   : Number of scales to generate (recommended: 30–40)
        s_min        : Minimum scale (default: max(16, order + 4))
        s_max_ratio  : Maximum scale as a proportion of the sequence length (default: 0.25)

    Returns:
        np.ndarray(int): Recommended array of scale windows
    """
    if s_min is None:
        s_min = max(16, order + 4)

    s_max = int(x_len * s_max_ratio)
    if s_max <= s_min + 3:
        raise ValueError(f"Sequence is too short to construct a valid scale range: s_max={s_max} <= s_min={s_min}")

    lag = np.logspace(np.log10(s_min), np.log10(s_max), num=num_scales)
    lag = np.unique(np.round(lag).astype(int))
    lag = lag[(lag >= s_min) & (lag <= s_max)]

    return lag


class CFA1DMFS:
    '''
    data: input data (time serie)
    q_list: range of q values
    with_progress：if show progress bar
    '''
    def __init__(self, data, q_list= np.linspace(-5, 5, 51), with_progress= True ):
        self.m_data = data    
        self.m_q_list = q_list
        self.m_with_progress = with_progress
    """
    Calculate the multifractal spectrum (MFS), including generalized Hurst exponent h(q),
    mass exponent τ(q), singularity strength α(q), multifractal spectrum f(α), 
    and generalized dimension D(q).

    Parameters:
        lag_list : list or np.ndarray, optional
            Custom list of scale windows. If None, automatically generate recommended scales.
        order    : int, optional
            Polynomial fitting order for DFA. Default is 2.

    Returns:
        pd.DataFrame containing q, h(q), τ(q), α(q), f(α), and D(q) multifractal metrics.
    """
    def get_mfs(self, lag_list = None, order = 2):
        
        df_result = pd.DataFrame()
        if lag_list == None:
            lag = recommended_lag(len(self.m_data))
        else:
            lag = lag_list
            
        q = self.m_q_list
        lag_used, dfa = MFDFA(self.m_data, lag, order, q=q, stat=False)
        
        if dfa.ndim == 1:
            dfa = dfa[:, np.newaxis]
        
        if dfa.shape[1] != len(q):
            print(f"Warning: dfa.shape[1]={dfa.shape[1]} != len(q)={len(q)}, trimming q.")
            q = q[:dfa.shape[1]]

        df_result['q'] = pd.Series(q)
        log_s = np.log2(lag_used)
        Hq = []
        
        for i in tqdm(range(len(q))):
            fq = dfa[:, i]
            mask = fq > 0
            if np.sum(mask) < 2:
                Hq.append(np.nan)
                continue
            coeffs = np.polyfit(log_s[mask], np.log2(fq[mask]), 1)
            Hq.append(coeffs[0])
        
        Hq = np.array(Hq)
        
        df_result['h(q)'] = pd.Series(Hq)

        tau_q = q * Hq - 1
        alpha = np.gradient(tau_q, q)
        f_alpha = q * alpha - tau_q
        
        df_result['t(q)'] = pd.Series(tau_q)
        df_result['a(q)'] = pd.Series(alpha)
        df_result['f(a)'] = pd.Series(f_alpha)
        
        Dq = np.full_like(q, np.nan)
        non_q1_mask = (q != 1)
        
        Dq[non_q1_mask] = tau_q[non_q1_mask] / (q[non_q1_mask] - 1)
        
        # Optional: approximate D(1) at q = 1 using the average of neighboring values
        if 1.0 in q:
            idx_q1 = np.where(q == 1.0)[0][0]
            # Approximate D(1) using np.gradient or interpolation
            dq_left = Dq[idx_q1 - 1]
            dq_right = Dq[idx_q1 + 1]
            Dq[idx_q1] = (dq_left + dq_right) / 2 # Simple average used for approximation
            
        df_result['d(q)'] = pd.Series(Dq)
        return df_result
        
    '''Plot multifractal spectrum'''
    def plot(self, df_mfs):
        fig, axs = plt.subplots(2, 3, figsize=(16, 8))
        
        #  H(q) vs. q
        df_tmp = df_mfs.groupby("q").mean().reset_index().sort_values(by = "q").reset_index()
        sns.lineplot(data=df_tmp, x='q', y='h(q)', ax=axs[0, 0])
        axs[0, 0].set(xlabel='$q$', ylabel=r'$H(q)$', title=r'$H(q)$ vs. $q$')
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
    x = np.cumsum(np.random.randn(5000))
    q = np.linspace(-5, 5, 21)
    mfs = CFA1DMFS(x)
    df_mfs = mfs.get_mfs()
    mfs.plot(df_mfs)

if __name__ == "__main__":
    main()
