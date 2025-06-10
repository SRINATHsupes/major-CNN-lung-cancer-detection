# major-CNN-lung-cancer-detection

#Lung Cancer Detection using CNN (EfficientNet)

This deep learning project uses **EfficientNet-based Convolutional Neural Network (CNN)** to classify lung cancer into **three classes**:

- ACA (Adenocarcinoma)
-  N (Normal)
-  SCC (Squamous Cell Carcinoma)

---

## Model Overview

The model architecture is built on top of **EfficientNet**, a state-of-the-art CNN that balances accuracy and efficiency. It uses:

-  **ReLU** activation in hidden layers
-  **Softmax** activation in the final layer
-  **Categorical Crossentropy** as the loss function for multi-class classification

---

##  Dataset

The dataset consists of labeled lung cancer images divided into three categories:
- **ACA** - Adenocarcinoma (malignant)
- **SCC** - Squamous Cell Carcinoma (malignant)
- **N** - Normal lung tissue
- dataset from here ((https://drive.google.com/file/d/1i0PC5cysr-WSNisnkQngw6bcR7gN_uIM/view?usp=drive_link))


----

## Tools & Libraries

- Python
- TensorFlow / Keras
- NumPy, Matplotlib, Seaborn
- EfficientNet from `tensorflow.keras.applications`

---

##  Training Details

| Metric               | Value               |
|----------------------|---------------------|
| Optimizer            | Adam                |
| Loss Function        | Categorical Crossentropy |
| Final Activation     | Softmax             |
| Hidden Activation    | ReLU                |
| Batch Size           | 32                  |
| Epochs               | 25–50               |
| Evaluation Metrics   | Accuracy, Precision, Recall, F1-score |

---

## Sample Results

- **Training Accuracy:** ~96%
- **Validation Accuracy:** ~92%
- **Test F1-score:** 0.91



---

##  Example Predictions

| Actual | Predicted | Confidence |
|--------|-----------|------------|
| ACA    | ACA       | 98%        |
| SCC    | N         | 96%        |
| N      | N         | 99%        |

---
