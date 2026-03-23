# FreeAeon-Fractal

**FreeAeon-Fractal** is a Python toolkit for computing **Multifractal Spectra**, **Fractal Dimensions**, **Fractal Lacunarity** and **Fourier Spectra** of images or series.

## 📦 Installation

Install via pip:

```bash
pip install FreeAeon-Fractal
```

> 💡 Requires Python 3.6+ and OpenCV (`cv2`) support.

## 🖼 Usage

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

### Lacunarity analysis of an image

```bash
python python demo.py --mode=lacunarity --image=./images/fractal.png
```

Example:

![Fractal Dimension Input](https://github.com/jim-xie-cn/FreeAeon-Fractal/raw/main/images/lacunarity.png)

### Fourier analysis of an image

```bash
python demo.py --mode fourier --image ./images/face.png
```

Example:

![Fractal Dimension Input](https://github.com/jim-xie-cn/FreeAeon-Fractal/raw/main/images/fourier.png)

### Calculate the **Multifractal Spectrum** of a Series

```bash
python demo.py --mode series
```

Example:

![Fractal Dimension Input](https://github.com/jim-xie-cn/FreeAeon-Fractal/raw/main/images/series.png)

### Parameters

- `--image`: Path to the input image  
- `--mode`: Analysis mode:  
  - `fd` – Fractal Dimension  
  - `mfs` – Multifractal Spectrum (default)
  - `lacunarity` - Lacunarity analysis
  - `fourier` - Fourier analysis
  - `series` - Multifractal Spectrum for Series analysis

## Use GPU to speed up

```bash
from FreeAeonFractal.FAImageDimensionGPU import CFAImageDimensionGPU as CFAImageDimension
from FreeAeonFractal.FA2DMFSGPU import CFA2DMFSGPU as CFA2DMFS
```
## 📚 使用文档

完整使用说明、详细参数介绍及进阶示例请参考：

🔗 https://github.com/jim-xie-cn/FreeAeon-Fractal/blob/dev/docs/markdown/en/index.md

## 📁 Directory Structure

```
FreeAeon-Fractal/
├── FreeAeonFractal/      # Core module
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

Yin Jie

📧 E-Mail: yinjiejspi@163.com

Cindy Ma

📧 E-Mail: 453303661@qq.com

Wenjing Zhang

📧 E-Mail: 634676988@qq.com

Danny Zhang

📧 E-Mail: zhyzxsw@126.com

---

## 🧠 Citation

If you use this project in academic work, please cite it as:

> Jim Xie, *FreeAeon-Fractal: A Python Toolkit for Fractal and Multifractal Image Analysis*, 2025.  
> GitHub Repository: https://github.com/jim-xie-cn/FreeAeon-Fractal
