# Fractal Sample Generator - CFASample

## Application Scenarios

The `CFASample` class generates classic fractal patterns based on Iterated Function Systems (IFS). Main application scenarios include:

- **Algorithm Testing**: Generate ground-truth fractals with known dimensions
- **Education**: Demonstrate fractal geometry concepts
- **Benchmarking**: Compare analysis methods on known fractal structures
- **Artistic Creation**: Generate fractal artwork
- **Data Augmentation**: Create synthetic fractal textures

## Usage Examples

### Generate Classic Fractals

```python
import numpy as np
import matplotlib.pyplot as plt
from FreeAeonFractal.FASample import CFASample
from FreeAeonFractal.FAVisual import CFAVisual

# 1D: Cantor Set (dimension ≈ 0.63)
points_1d = CFASample.get_Cantor_Set(iterations=256)

# 2D: Sierpinski Triangle (dimension ≈ 1.58)
points_2d = CFASample.get_Sierpinski_Triangle(iterations=1024)

# 2D: Barnsley Fern (dimension ≈ 1.67)
points_fern = CFASample.get_Barnsley_Fern(iterations=4096)

# 3D: Menger Sponge (dimension ≈ 2.73)
points_3d = CFASample.get_Menger_Sponge(iterations=10240)

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(141); CFAVisual.plot_1d_points(points_1d, ax1); ax1.set_title("Cantor Set")
ax2 = fig.add_subplot(142); CFAVisual.plot_2d_points(points_2d, ax2); ax2.set_title("Sierpinski Triangle")
ax3 = fig.add_subplot(143); CFAVisual.plot_2d_points(points_fern, ax3); ax3.set_title("Barnsley Fern")
ax4 = fig.add_subplot(144, projection='3d'); CFAVisual.plot_3d_points(points_3d, ax4); ax4.set_title("Menger Sponge")
plt.tight_layout()
plt.show()
```

### Convert Points to Image

```python
from FreeAeonFractal.FASample import CFASample

points = CFASample.get_Sierpinski_Triangle(iterations=4096)
image = CFASample.get_image_from_points(points, img_size=(512, 512))

# Now use for fractal analysis
from FreeAeonFractal.FAImageFD import CFAImageFD
fd_bc = CFAImageFD(image).get_bc_fd()
print("Sierpinski FD (BC):", fd_bc['fd'])  # should be ≈ 1.58
```

### Custom IFS

```python
import numpy as np
from FreeAeonFractal.FASample import CFASample

# Custom transformation matrices and probabilities
trans_matrix = np.array([
    [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],    # scale by 0.5
    [[0.5, 0.0, 1.0], [0.0, 0.5, 0.0]],    # scale + translate x
    [[0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],    # scale + translate xy
])
trans_probability = np.array([0.33, 0.33, 0.34])
init_point = np.array([0.0, 0.0])

points = CFASample.generate(init_point, iterations=2048, trans_matrix=trans_matrix, trans_probability=trans_probability)
```

### Installation

```bash
pip install FreeAeon-Fractal
```

## Class Description

### CFASample

**Description**: IFS (Iterated Function System) fractal generator. All generation methods use randomized iterated affine transformations.

#### Methods

##### generate(init_point, iterations, trans_matrix, trans_probability) [static]

**Description**: Core IFS generator. Applies randomly selected affine transformations iteratively.

**Parameters**:
- `init_point` (ndarray): Starting point in the appropriate dimension
- `iterations` (int): Number of IFS iterations
- `trans_matrix` (ndarray): Array of affine transformation matrices; shape `(n_transforms, ndim, ndim+1)`
- `trans_probability` (ndarray): Probability of selecting each transformation (must sum to 1)

**Return Value**: `(iterations, ndim)` array of generated points

##### get_Cantor_Set(init_point=np.array([0.0]), iterations=256) [static]

**Description**: Generate 1D Cantor Set points.

**Return Value**: (iterations,) point array

**Theoretical dimension**: ≈ 0.6309 (log 2 / log 3)

##### get_Sierpinski_Triangle(init_point=np.array([0.0, 0.0]), iterations=256) [static]

**Description**: Generate 2D Sierpinski Triangle points.

**Return Value**: (iterations, 2) point array

**Theoretical dimension**: ≈ 1.585 (log 3 / log 2)

##### get_Barnsley_Fern(init_point=np.array([0.0, 0.0]), iterations=4096)

**Description**: Generate 2D Barnsley Fern points.

**Return Value**: (iterations, 2) point array

**Theoretical dimension**: ≈ 1.67

##### get_Menger_Sponge(init_point=np.array([0.0, 0.0, 0.0]), iterations=10240) [static]

**Description**: Generate 3D Menger Sponge points using 20 contraction maps (3D cube minus the 7 center/edge pieces).

**Return Value**: (iterations, 3) point array

**Theoretical dimension**: ≈ 2.727 (log 20 / log 3)

##### get_image_from_points(points, img_size=(512, 512), margin=0.05) [static]

**Description**: Convert 2D IFS point cloud to a binary uint8 image.

**Parameters**:
- `points` (ndarray): (N, 2) point array
- `img_size` (tuple): Output image dimensions `(H, W)`
- `margin` (float): Fractional margin added around point bounds

**Return Value**: `(H, W)` uint8 image with value 255 at occupied pixels

## Fractal Dimensions

| Fractal | Dimension | Method |
|---------|-----------|--------|
| Cantor Set | ≈ 0.63 | 1D box-counting |
| Sierpinski Triangle | ≈ 1.58 | 2D box-counting |
| Barnsley Fern | ≈ 1.67 | 2D box-counting |
| Menger Sponge | ≈ 2.73 | 3D box-counting |

## Important Notes

1. **Iteration Count**: More iterations produce denser, more accurate fractal approximations; 1000+ is recommended for analysis
2. **Image Size**: For fractal dimension analysis, 256×256 or larger images give more reliable scale ranges
3. **IFS Convergence**: The IFS chaos game converges regardless of starting point after a burn-in period; the first few hundred points can be discarded if needed
4. **get_Barnsley_Fern**: Note this method is not decorated as `@staticmethod` in the current code; call via `CFASample.get_Barnsley_Fern()`

## References

- Barnsley, M. F. (1988). *Fractals Everywhere*. Academic Press.
- Mandelbrot, B. B. (1982). *The Fractal Geometry of Nature*. Freeman.
