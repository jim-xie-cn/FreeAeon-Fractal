# FreeAeon-Fractal

**FreeAeon-Fractal** is a Python toolkit for computing **Multifractal Spectra** and **Fractal Dimensions** of images.

## 📦 Installation

Install via pip:

```bash
pip install FreeAeon-Fractal
```

> 💡 Requires Python 3.6+ and OpenCV (`cv2`) support.

## 🖼 Usage

### Calculate the **Multifractal Spectrum** of an image

```bash
python demo.py --image ./images/face.png --mode mfs
```

Example:

![Multifractal Spectrum Input](https://github.com/jim-xie-cn/FreeAeon-Fractal/raw/main/images/mfs.png)

### Calculate the **Fractal Dimensions** (Box-Counting, DBC, SDBC) of an image

```bash
python demo.py --image ./images/fractal.png --mode fd
```

Example:

![Fractal Dimension Input](https://github.com/jim-xie-cn/FreeAeon-Fractal/raw/main/images/fd.png)

### Fourier analysis of an image

```bash
python demo.py --image ./images/face.png  --mode fourier
```

Example:

![Fractal Dimension Input](https://github.com/jim-xie-cn/FreeAeon-Fractal/raw/main/images/fourier.png)


### Parameters

- `--image`: Path to the input image  
- `--mode`: Analysis mode:  
  - `fd` – Fractal Dimension  
  - `mfs` – Multifractal Spectrum (default)
  - `fourier` - Fourier analysis

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

---

## 🧠 Citation

If you use this project in academic work, please cite it as:

> Jim Xie, *FreeAeon-Fractal: A Python Toolkit for Fractal and Multifractal Image Analysis*, 2025.  
> GitHub Repository: https://github.com/jim-xie-cn/FreeAeon-Fractal
