# FreeAeon-Fractal

**FreeAeon-Fractal** is a Python toolkit for computing **Multifractal Spectra**, **Fractal Dimensions**, **Fractal Lacunarity** and **Fourier Spectra** of images or series.  🔗 [More](https://github.com/jim-xie-cn/FreeAeon-Fractal/wiki)

### Key Features

- 🎯 **Multifractal Spectrum Analysis**: Supports 2D images and 1D time series
- 📏 **Fractal Dimension Calculation**: BC, DBC, and SDBC methods
- 🔍 **Lacunarity Analysis**: Quantify spatial heterogeneity
- 🌊 **Fourier Analysis**: Frequency domain analysis and filtering
- ⚡ **GPU Acceleration**: Optional GPU support for faster computation
- 📊 **Visualization**: Built-in rich visualization capabilities

## Application Domains

### Medical Imaging
- **Tissue Complexity**: Quantify tissue structure via fractal dimension
- **Heterogeneity Analysis**: Reveal lesion characteristics via multifractal spectrum
- **Texture Classification**: Image classification based on fractal features

### Materials Science
- **Surface Morphology**: Describe surface roughness via fractal dimension
- **Porous Structure**: Analyze internal structure via lacunarity
- **Fracture Analysis**: Identify fracture patterns via multifractal features

### Financial Analysis
- **Price Fluctuations**: Analyze stock prices via multifractal spectrum
- **Risk Assessment**: Quantify risk based on fractal features
- **Market Prediction**: Long-range correlation analysis

### Earth Sciences
- **Terrain Analysis**: Describe terrain complexity via fractal dimension
- **Vegetation Distribution**: Quantify vegetation coverage via lacunarity
- **Climate Series**: Multifractal analysis of time series

### Image Processing
- **Texture Classification**: Texture recognition based on fractal features
- **Image Segmentation**: ROI extraction based on multifractal analysis
- **Quality Assessment**: Image complexity evaluation

## 📦 Installation

Install via pip:

```bash
pip install FreeAeon-Fractal
```

> 💡 Requires Python 3.6+ and OpenCV (`cv2`) support.


## 🖼 Usage

### Get sample code

```bash
git clone https://github.com/jim-xie-cn/FreeAeon-Fractal.git
cd FreeAeon-Fractal
```

### Calculate the **Multifractal Spectrum** of an image

```bash
python demo.py --mode mfs --image ./images/face.png
```

Example:

![Multifractal Spectrum Input](https://github.com/jim-xie-cn/FreeAeon-Fractal/raw/main/images/mfs.png)

### Calculate the **Fractal Dimensions** (Box-Counting, DBC, SDBC) of an image

```bash
python demo.py --mode fd --image ./images/fractal.png
```

Example:

![Fractal Dimension Input](https://github.com/jim-xie-cn/FreeAeon-Fractal/raw/main/images/fd.png)

### Calculate the **Local Multifractal $alpha$-map** of an image

```bash
python demo.py --mode alpha --image ./images/face.png
```

Example:

![Local Multifractal $alpha$-map Input](https://github.com/jim-xie-cn/FreeAeon-Fractal/raw/main/images/face.png)


### Lacunarity analysis of an image

```bash
python demo.py --mode=lacunarity --image=./images/fractal.png
```

Example:

![Lacunarity Input](https://github.com/jim-xie-cn/FreeAeon-Fractal/raw/main/images/lacunarity.png)

### Fourier analysis of an image

```bash
python demo.py --mode fourier --image ./images/face.png
```

Example:

![Fourier Input](https://github.com/jim-xie-cn/FreeAeon-Fractal/raw/main/images/fourier.png)

### Calculate the **Multifractal Spectrum** of a Series

```bash
python demo.py --mode series
```

Example:

![Series Input](https://github.com/jim-xie-cn/FreeAeon-Fractal/raw/main/images/series.png)

### Parameters

- `--image`: Path to the input image  
- `--mode`: Analysis mode:  
  - `fd` – Fractal Dimension  
  - `mfs` – Multifractal Spectrum (default)
  - `lacunarity` - Lacunarity analysis
  - `fourier` - Fourier analysis
  - `series` - Multifractal Spectrum for Series analysis

## Use GPU to speed up

```python
from FreeAeonFractal.FAImageFDGPU import CFAImageFDGPU as CFAImageFD
from FreeAeonFractal.FAImageMFSGPU import CFAImageMFSGPU as CFAImageMFS
from FreeAeonFractal.FAImageLACGPU import CFAImageLACGPU as CFAImageLAC
```

## 📚 User Guide
See the following for complete usage, parameter details, and advanced examples:

### Guide (Markdown) - 🐙 [GitHub View]
> 💡 Best for viewing directly on GitHub.

🔗 [English Edition](https://github.com/jim-xie-cn/FreeAeon-Fractal/blob/main/docs/markdown/en/index.md)
🔗 [Chinese Edition](https://github.com/jim-xie-cn/FreeAeon-Fractal/blob/main/docs/markdown/zh/index.md)

### Guide (HTML) - 🌐 [Browser View]
> 💡 Recommended for direct reading in browser.

🔗 [English Edition](https://jim-xie-cn.github.io/FreeAeon-Fractal/docs/html/en/index.html)
🔗 [Chinese Edition](https://jim-xie-cn.github.io/FreeAeon-Fractal/docs/html/zh/index.html)

## 📁 Directory Structure

```
FreeAeon-Fractal/
├── FreeAeonFractal/      # Core module
│   ├── FAImageMFS.py     # 2D multifractal spectrum
│   ├── FAImageMFSGPU.py  # 2D multifractal spectrum (GPU)
│   ├── FASeriesMFS.py    # 1D multifractal spectrum
│   ├── FAImageFD.py      # Fractal dimension
│   ├── FAImageFDGPU.py   # Fractal dimension (GPU)
│   ├── FAImageLAC.py     # Lacunarity
│   ├── FAImageLACGPU.py  # Lacunarity (GPU)
│   ├── FAImageFourier.py # Fourier analysis
│   ├── FAImage.py        # Image utilities
│   ├── FASample.py       # Fractal sample generator
│   └── FAVisual.py       # Visualization tools
├── demo.py               # CLI interface
├── images/               # Example images
├── requirements.txt
├── setup.py
└── README.md
```

## 📄 License

This project is licensed under the MIT License. See [LICENSE](https://github.com/jim-xie-cn/FreeAeon-Fractal/blob/main/LICENSE) for details.

## ✍️ Author

Jim Xie  

📧 E-Mail: jim.xie.cn@outlook.com, xiewenwei@sina.com

🔗 GitHub: https://github.com/jim-xie-cn/FreeAeon-Fractal

---

## 🧠 Citation

If you use this project in academic work, please cite it as:

> Jim Xie, *FreeAeon-Fractal: A Python Toolkit for Fractal and Multifractal Image Analysis*, 2025.  
> GitHub Repository: https://github.com/jim-xie-cn/FreeAeon-Fractal
