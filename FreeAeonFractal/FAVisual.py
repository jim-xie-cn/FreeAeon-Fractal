'''
Show image and points
'''
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import logging

class CFAVisual(object):

    '''show 1D points'''
    @staticmethod    
    def plot_1d_points(points,ax = plt ):
        ax.scatter(points, np.zeros_like(points), s=1)
        ax.get_yaxis().set_visible(False)  

    '''show 2D points'''
    @staticmethod    
    def plot_2d_points(points,ax = plt ):
        ax.scatter(points[:, 0], points[:, 1], s=1, c='red', marker='.')

    '''show 2D image'''
    @staticmethod    
    def plot_2d_image(image,cmap='gray',ax = plt):
        ax.imshow(image, cmap=cmap)

    '''show 3D points'''
    @staticmethod    
    def plot_3d_points(points,ax = None ):
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='red', marker='.',s=1)
    
    '''show 3D image'''
    @staticmethod
    def plot_3d_image(img,ax = plt):
        x,y,z = img[:,0],img[:,1],img[:,2]
        if img.ndim == 2 and img.shape[1] >= 4:
            value = img[:,3]
        else:
            value = np.ones_like(x)
        scatter = ax.scatter(x, y, z, s=value*10, marker='.', cmap='viridis')

def main():
    pass

if __name__ == "__main__":
    main()
