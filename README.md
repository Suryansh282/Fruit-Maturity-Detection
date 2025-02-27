# Fruit Maturity Detection Using Hyperspectral Imaging

This project leverages **hyperspectral imaging** and **deep learning** to classify fruit maturity stages (**unripe, ripe, overripe**). Using **PyTorch, ResNet-50, and Transformer Encoders**, the model extracts spatial and spectral features to improve classification accuracy.

---

## Features
- **Hyperspectral Data Processing**: Handles high-dimensional spectral data efficiently.
- **Deep Learning Architecture**: Combines **ResNet-50** for spatial features and **Transformers** for spectral dependencies.
- **Custom Data Augmentation**: Implements **3D transformations** for robust training.
- **Scalable & Reproducible**: Supports **experiment tracking and model deployment**.

---

## Installation
### Prerequisites
Ensure you have **Python 3.8+** and install dependencies:

```bash
pip install torch torchvision torchaudio numpy pandas matplotlib seaborn scikit-learn
```
## Dataset
- **Input**: Hyperspectral images (**64x64 pixels, 249 spectral bands per image**).
- **Labels**: Three categories - **Unripe, Ripe, Overripe**.
- **Format**: Dataset stored as a **Tensor dataset** (`.pt` file).

---

## Model Architecture
1. **ResNet-50 Backbone** - Extracts spatial features.
2. **Transformer Encoder** - Captures inter-band spectral relationships.
3. **Classification Layers** - Predicts ripeness category.

---

## Usage
### Train the Model
```bash
python train.py --epochs 100 --batch_size 16 --lr 0.001

python evaluate.py --model_path saved_model.pth

```

## Results
- **Accuracy**: 79.14%
- **Precision**: 82.31%
- **F1 Score**: 77.9%

---

## Future Work
- Improve accuracy with **better spectral augmentations**.
- Optimize inference for **edge deployment**.
- Extend to other fruit categories beyond **mangoes**.

---

## Contributors
- **Aditya Raj**
- **Kuldeep Chaudhary**
- **Suryansh Singh**

---

## License
This project is licensed under the **MIT License**.

