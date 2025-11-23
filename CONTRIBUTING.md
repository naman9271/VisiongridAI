# Contributing to VisionGrid AI 🤝

Thank you for your interest in contributing to VisionGrid AI! This document provides guidelines and instructions for setting up the project and contributing effectively.

---

## 📋 Table of Contents
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Contributing Guidelines](#contributing-guidelines)
- [Testing](#testing)
- [Code Style](#code-style)
---

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python**: 3.8 or higher
- **Git**: For version control
- **CUDA Toolkit** (optional): For GPU acceleration (CUDA 11.0+)
- **pip**: Python package manager

### System Requirements

**Minimum Requirements:**
- RAM: 4GB (8GB recommended)
- Storage: 5GB free space
- CPU: Multi-core processor

**For GPU Inference:**
- NVIDIA GPU with CUDA support
- 4GB+ VRAM

---

## 💻 Development Setup

### 1. Fork and Clone the Repository

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/Dense-Object-Detection.git
cd Dense-Object-Detection

# Add upstream remote
git remote add upstream https://github.com/naman9271/VisiongridAI.git
```

### 2. Create a Virtual Environment

**Using venv (recommended):**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n visiongrid python=3.8
conda activate visiongrid
```

### 3. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install YOLOv5 dependencies
cd yolov5
pip install -r requirements.txt
cd ..

# Install development dependencies (optional)
pip install pytest black flake8 pylint
```

### 4. Download Model Weights

```bash
# Create weights directory
mkdir -p weights

# Download pre-trained weights from Google Drive
# Link: https://drive.google.com/file/d/1BRlXZD9MqYAYYnciMRQ50Mht9kBncf0l/view?usp=sharing
# Place best.pt in the weights/ directory
```

### 5. Verify Installation

```bash
# Test Python imports
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

# Test CUDA availability (if GPU available)
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### 6. Run the Application

```bash
# Launch Streamlit web application
streamlit run app.py

# Application will open at http://localhost:8501
```

---

## 📁 Project Structure

```
Dense-Object-Detection/
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── packages.txt               # System packages (for Streamlit Cloud)
├── README.md                  # Project documentation
├── CONTRIBUTING.md            # This file
├── CITATION.cff              # Citation information
├── LICENSE                   # MIT License
├── assets/                   # Images and visualizations
│   ├── results.png
│   ├── confusion_matrix.png
│   └── inference examples
├── test/                     # Test results for different models
│   ├── pt_model/
│   ├── tf_model/
│   ├── tflite_fp16_model/
│   ├── tflite_fp32_model/
│   └── tflite_int8_model/
├── weights/                  # Model weights (not in repo)
│   └── best.pt
└── yolov5/                  # YOLOv5 submodule
    ├── detect.py            # Inference script
    ├── train.py             # Training script
    ├── val.py               # Validation script
    ├── export.py            # Model export
    ├── models/              # Model architectures
    ├── utils/               # Utility functions
    └── data/                # Dataset configurations
```

---

## 🤝 Contributing Guidelines

### Workflow

1. **Create a new branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```

2. **Make your changes** following the code style guidelines

3. **Test your changes** thoroughly

4. **Commit your changes** with clear, descriptive messages:
   ```bash
   git add .
   git commit -m "feat: add ONNX export functionality"
   # or
   git commit -m "fix: resolve inference memory leak"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub

### Commit Message Convention

Use conventional commits for clear history:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

**Examples:**
```
feat: add TensorRT export support
fix: correct bounding box scaling on mobile devices
docs: update installation instructions for Windows
refactor: optimize image preprocessing pipeline
```

---

## 🧪 Testing

### Running Inference Tests

```bash
# Test on sample image
python yolov5/detect.py --weights weights/best.pt --source test_image.jpg --conf 0.25

# Test on video
python yolov5/detect.py --weights weights/best.pt --source test_video.mp4
```

### Testing Web Application

```bash
# Run Streamlit app in test mode
streamlit run app.py --server.headless true

# Test with different configurations
# Upload various test images through the interface
```

### Model Validation

```bash
# Run validation on test set (requires SKU-110K dataset)
python yolov5/val.py --data data/sku110k.yaml --weights weights/best.pt --img 640
```

### Unit Tests (if applicable)

```bash
# Run pytest
pytest tests/

# Run specific test file
pytest tests/test_inference.py
```

---

## 💅 Code Style

### Python Code Style

We follow **PEP 8** guidelines with some modifications:

- **Line length**: 100 characters (not 79)
- **Indentation**: 4 spaces
- **Quotes**: Single quotes for strings (unless double quotes avoid escaping)

### Formatting Tools

```bash
# Format code with Black
black app.py --line-length 100

# Check with flake8
flake8 app.py --max-line-length 100

# Check with pylint
pylint app.py
```

---

## 📄 License

By contributing to VisionGrid AI, you agree that your contributions will be licensed under the MIT License.

---

## 🙏 Acknowledgments

Thank you for contributing to VisionGrid AI! Your efforts help make this project better for everyone.

**Special thanks to all contributors who have helped improve this project!**

---

<div align="center">

**Questions?** Feel free to reach out via [GitHub Issues](https://github.com/naman9271/VisiongridAI/issues) or email!

**Happy Contributing! 🎉**

</div>
