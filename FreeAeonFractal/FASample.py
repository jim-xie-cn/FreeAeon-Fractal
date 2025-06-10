'''
Generate normal fractal images
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from .FAVisual import CFAVisual

# Cantor Set
g_m_Cantor_Set = np.array([[[1/3, 0]],[[1/3, 2/3]]])
g_p_Cantor_Set = np.array([0.5, 0.5])

#Sierpinski Triangle
g_m_Sierpinski_Triangle = np.array([[[+0.50,+0.00,+0.00],[+0.00,+0.50,+0.00]],
                  [[+0.50,+0.00,+1.00],[+0.00,+0.50,+0.00]],
                  [[+0.50,+0.00,+0.50],[+0.00,+0.50,+0.50]]])
g_p_Sierpinski_Triangle = np.array([0.33, 0.33, 0.34])

#Barnsley Fern
g_m_Barnsley_Fern = np.array([[[+0.00,+0.00,+0.00],[+0.00,+0.16,+0.00]],
                  [[+0.85,+0.04,+0.00],[+0.04,+0.85,+1.60]],
                  [[+0.20,-0.26,+0.50],[+0.23,+0.22,+1.60]],
                  [[-0.15,+0.28,+0.00],[+0.26,+0.24,+0.44]]])
g_p_Barnsley_Fern = np.array([0.25, 0.25, 0.25,0.25])

#4D test
g_m_4 = np.array([[[+0.50,+0.00,+0.00,+0.00,+0.20],[+0.00,+0.50,+0.00,+0.00,+0.30],[+0.00,+0.50,+0.50,+0.00,+0.00],[+0.00,+0.50,+0.50,+0.00,+0.00]],
                  [[+0.50,+0.00,+0.00,+1.00,+0.30],[+0.00,+0.50,+0.50,+0.00,+0.20],[+0.00,+0.50,+0.50,+0.00,+0.00],[+0.00,+0.50,+0.50,+0.00,+0.00]],
                  [[+0.50,+0.00,+0.00,+0.50,+0.40],[+0.00,+0.50,+0.00,+0.50,+0.20],[+0.00,+0.50,+0.50,+0.00,+0.00],[+0.00,+0.50,+0.50,+0.00,+0.00]],
                  [[+0.10,+0.20,+0.30,+0.40,+0.50],[+0.50,+0.60,+0.70,+0.80,+0.10],[+0.90,+0.00,+0.00,+0.00,+0.00],[+0.90,+0.00,+0.00,+0.00,+0.00]]
                  ])
g_p_4 = np.array([0.25, 0.25, 0.25,0.25])

class CFASample(object):

    '''generate point'''
    @staticmethod  
    def generate(init_point, iterations,trans_matrix,trans_probability):
        def align_matrix(mat):
            tmp = np.zeros(mat.shape[1])
            tmp[-1] = 1
            return np.vstack((mat, tmp))
        
        result_points = []
        current_point = init_point
        for _ in tqdm(range(iterations)):
            chosen_transform = np.random.choice(len(trans_probability), p=trans_probability)
            matrix = trans_matrix[chosen_transform]
            affine_matric = align_matrix( matrix )
            current_point = np.dot(affine_matric, np.concatenate([current_point, [1]]))[:-1]
            result_points.append(current_point)

        return np.array(result_points)
    
    '''Cantor Set which dimension is 0.6309'''
    @staticmethod
    def get_Cantor_Set(init_point = np.array([0.0]), iterations = 256 ):
        return CFASample.generate(init_point,iterations,g_m_Cantor_Set,g_p_Cantor_Set)

    '''Sierpinski Triangle which dimension is 1.58'''
    @staticmethod
    def get_Sierpinski_Triangle(init_point = np.array([0.0, 0.0]), iterations = 256 ):
        return CFASample.generate(init_point,iterations,g_m_Sierpinski_Triangle,g_p_Sierpinski_Triangle)
        
    '''Barnsley Fern which dimension is 1.67'''
    def get_Barnsley_Fern(init_point = np.array([0.0, 0.0]), iterations = 4096):
        return CFASample.generate(init_point,iterations,g_m_Barnsley_Fern,g_p_Barnsley_Fern)

    '''Menger Sponge which dimension is 1.67'''
    @staticmethod
    def get_Menger_Sponge(init_point = np.array([0.0, 0.0, 0.0]), iterations = 10240):
        def get_menger_ifs_matrices():
            scale = 1 / 3
            positions = []
            for x in range(3):
                for y in range(3):
                    for z in range(3):
                        # 剔除中间块和中间轴（共7个）
                        if (x == 1 and y == 1) or (x == 1 and z == 1) or (y == 1 and z == 1):
                            continue
                        positions.append((x, y, z))
            
            assert len(positions) == 20

            matrices = []
            for pos in positions:
                tx, ty, tz = np.array(pos) * scale
                A = np.array([
                    [scale,     0,     0, tx],
                    [    0, scale,     0, ty],
                    [    0,     0, scale, tz]
                ])
                matrices.append(A)
            
            matrices = np.array(matrices)  # shape: (20, 3, 4)
            probs = np.ones(len(matrices)) / len(matrices)
            return matrices, probs
        matrices, probs = get_menger_ifs_matrices()
        return CFASample.generate(init_point,iterations,matrices,probs)

    '''4D points'''
    @staticmethod
    def get_4D_Points(init_point = np.array([0.0, 0.0, 0.0,0.0]), iterations = 4096):
        return CFASample.generate(init_point,iterations,g_m_4,g_p_4)
    
    '''
    convert points to an 2d image
    '''
    @staticmethod
    def get_image_from_points(points, img_size=(512, 512), margin=0.05):
        min_xy = points.min(axis=0)
        max_xy = points.max(axis=0)
        range_xy = max_xy - min_xy
        min_xy = min_xy - range_xy * margin
        max_xy = max_xy + range_xy * margin

        norm_points = (points - min_xy) / (max_xy - min_xy)

        px = (norm_points[:, 0] * (img_size[1] - 1)).astype(int)
        py = (norm_points[:, 1] * (img_size[0] - 1)).astype(int)

        img = np.zeros(img_size, dtype=np.uint8)

        img[py, px] = 255

        return img

def main():
    points_1 = CFASample.get_Cantor_Set()
    points_2 = CFASample.get_Sierpinski_Triangle()
    points_3 = CFASample.get_Barnsley_Fern()
    points_4 = CFASample.get_Menger_Sponge()
    points_5 = CFASample.get_4D_Points()
    
    image = CFASample.get_image_from_points(points_2)

    fig = plt.figure(figsize=(12, 6))

    # Sierpinski Triangle (2D)
    ax1 = fig.add_subplot(221)
    CFAVisual.plot_1d_points(points_1,ax1)
    ax1.set_title('Cantor Set')

    # Sierpinski Triangle (2D)
    ax2 = fig.add_subplot(222)
    CFAVisual.plot_2d_points(points_2,ax2)
    ax2.set_title('Sierpinski Triangle')

    # Barnsley Fern (2D)
    ax3 = fig.add_subplot(223)
    CFAVisual.plot_2d_points(points_3,ax3)
    ax3.set_title('Barnsley Fern')

    # Menger Sponge (3D)
    ax4 = fig.add_subplot(224, projection='3d')
    CFAVisual.plot_3d_points(points_4,ax4)
    ax4.set_title('Menger Sponge')

    plt.tight_layout()
    plt.show()

    print("4D points")
    print(points_5)

    CFAVisual.plot_2d_image(image)
    
    plt.show()
    
if __name__ == "__main__":
    main()
