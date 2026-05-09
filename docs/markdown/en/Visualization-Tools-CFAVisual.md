# Visualization Tools - CFAVisual

## Application Scenarios

The `CFAVisual` class provides visualization tools for fractal point sets and images. Main application scenarios include:

- **Fractal Teaching**: Visualize 1D/2D/3D fractal point sets
- **Research Verification**: Display generated fractals to validate IFS parameters
- **Image Display**: Show 2D analysis results side by side
- **3D Point Cloud**: Visualize 3D fractal structures

## Usage Examples

### 1D Point Visualization (Cantor Set)

```python
import matplotlib.pyplot as plt
from FreeAeonFractal.FAVisual import CFAVisual
from FreeAeonFractal.FASample import CFASample

# Generate Cantor Set points
points_1d = CFASample.get_Cantor_Set(iterations=256)

fig, ax = plt.subplots(figsize=(10, 2))
CFAVisual.plot_1d_points(points_1d, ax=ax)
ax.set_title("Cantor Set")
plt.show()
```

### 2D Point Visualization (Sierpinski Triangle)

```python
import matplotlib.pyplot as plt
from FreeAeonFractal.FAVisual import CFAVisual
from FreeAeonFractal.FASample import CFASample

points_2d = CFASample.get_Sierpinski_Triangle(iterations=256)

fig, ax = plt.subplots(figsize=(6, 6))
CFAVisual.plot_2d_points(points_2d, ax=ax)
ax.set_title("Sierpinski Triangle")
plt.show()
```

### 3D Point Visualization (Menger Sponge)

```python
import matplotlib.pyplot as plt
from FreeAeonFractal.FAVisual import CFAVisual
from FreeAeonFractal.FASample import CFASample

points_3d = CFASample.get_Menger_Sponge(iterations=10240)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
CFAVisual.plot_3d_points(points_3d, ax=ax)
ax.set_title("Menger Sponge")
plt.show()
```

### 2D Image Display

```python
import cv2
from FreeAeonFractal.FAVisual import CFAVisual

image = cv2.imread('./images/fractal.png', cv2.IMREAD_GRAYSCALE)
CFAVisual.plot_2d_image(image, cmap='gray')
```

### Installation

```bash
pip install FreeAeon-Fractal
```

## Class Description

### CFAVisual

**Description**: Static utility class for displaying fractal point sets (1D, 2D, 3D) and images. All methods accept an optional `ax` parameter for subplot integration.

#### Methods

##### plot_1d_points(points, ax=plt)

**Description**: Display 1D points on a horizontal scatter plot (y-axis hidden).

**Parameters**:
- `points` (ndarray): 1D array of point coordinates
- `ax`: Matplotlib axis or `plt` (default: `plt`)

**Use Case**: Cantor Set and other 1D fractal structures.

##### plot_2d_points(points, ax=plt)

**Description**: Display 2D points as a scatter plot.

**Parameters**:
- `points` (ndarray): (N, 2) array of (x, y) coordinates
- `ax`: Matplotlib axis or `plt`

**Use Case**: Sierpinski Triangle, Barnsley Fern, and other 2D IFS fractals.

##### plot_3d_points(points, ax=None)

**Description**: Display 3D points as a 3D scatter plot.

**Parameters**:
- `points` (ndarray): (N, 3) array of (x, y, z) coordinates
- `ax`: 3D Matplotlib axis (if None, a new figure is created)

**Use Case**: Menger Sponge and other 3D fractals.

##### plot_2d_image(image, cmap='gray', ax=plt)

**Description**: Display a 2D image using `imshow`.

**Parameters**:
- `image` (ndarray): 2D grayscale or 3D color image
- `cmap` (str): Colormap (default: `'gray'`)
- `ax`: Matplotlib axis or `plt`

##### plot_3d_image(img, ax=plt)

**Description**: Display a 3D point cloud from a structured (N, 4) array where the 4th column is the value for coloring.

**Parameters**:
- `img` (ndarray): (N, 3) or (N, 4) array; columns are (x, y, z[, value])
- `ax`: Matplotlib axis with 3D projection

## Important Notes

1. **Axes Parameter**: All methods accept `ax=plt` (uses pyplot directly) or an explicit axis object for subplot integration
2. **Point Density**: For dense fractals with >100K points, consider downsampling before display
3. **3D Projection**: `plot_3d_points` requires a 3D axis (`projection='3d'`); it creates one automatically if `ax=None`
