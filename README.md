# 👗 FashionMNIST Classification with PyTorch

This repository demonstrates a computer vision pipeline for classifying fashion items using the FashionMNIST dataset and PyTorch. It covers data loading, model architecture, training, evaluation, and testing—all within a clean, modular framework.

## 📌 Project Overview

The FashionMNIST dataset contains grayscale images of 10 fashion categories (e.g., T-shirt, sneaker, coat). This project builds a convolutional neural network (CNN) to classify these images with high accuracy and generalization.

## 🚀 Technologies Used

- **PyTorch**: Deep learning framework for model definition and training  
- **Torchvision**: Dataset loading and transformations  
- **NumPy & Matplotlib**: Data manipulation and visualization  
- **Jupyter Notebook**: Interactive experimentation  
- **Python Scripts**: Modular testing and utility functions  

## 🧠 Model Architecture

- **Input**: 28×28 grayscale image  
- **Conv Layers**: Two convolutional layers with ReLU and MaxPooling  
- **Fully Connected Layers**: Two dense layers with dropout  
- **Output**: Softmax over 10 classes  
- **Loss Function**: `CrossEntropyLoss`  
- **Optimizer**: `Adam` with learning rate tuning  

## 📊 Results

- ✅ **Test Accuracy**: ~89%  
- 📉 **Loss Curve**: Smooth convergence across epochs  
- 📈 **Confusion Matrix**: Strong separation across all 10 classes  
- 🔍 **Generalization**: Robust performance on unseen test samples  

## 📁 Repository Structure

```text
├── Data_fasionMNIST/FashionMNIST/raw        # Raw dataset files  
├── Models/                                  # Saved model checkpoints  
├── pytorch_fashionMNIT.ipynb                # Main notebook with full pipeline  
├── test.py                                  # Script for model testing  
├── requirement.txt                          # Python dependencies  
├── environment.yml                          # Conda environment setup  
├── .gitignore                               # Git ignore rules  
├── .gitattributes                           # Git attributes  
```
## 🛠️ How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/arvind207kumar/pytorch-computer-vision-fashion-classificatio.git
   cd pytorch-computer-vision-fashion-classificatio
2. **Install dependencies**:
   ```bash
   pip install -r requirement.txt
## 🧪 How to Test

Once the model is trained and saved, you can run the `test.py` script to evaluate its performance on the test dataset.

### 🔧 Run the test script:
```bash
python test.py


### 🔮 Future Work

This project lays the foundation for scalable computer vision pipelines. Potential future enhancements include:

- 📦 **Model Deployment**: Export trained model to ONNX or TorchScript for deployment on edge devices (e.g., Jetson Nano, Raspberry Pi)
- 🖼️ **Data Augmentation**: Integrate advanced augmentation techniques (e.g., CutMix, MixUp) to improve generalization
- 🧪 **Hyperparameter Tuning**: Use Optuna or Ray Tune for automated search across learning rates, batch sizes, and architectures
- 📊 **Dashboard Integration**: Visualize training metrics and predictions using Streamlit or Gradio
- 🧠 **Transfer Learning**: Fine-tune pretrained CNNs (e.g., ResNet18, EfficientNet) on FashionMNIST for faster convergence
- 📱 **Mobile Inference**: Convert model to CoreML or TensorFlow Lite for mobile deployment

> Contributions or suggestions are welcome—feel free to open an issue or pull request!
