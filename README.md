# Tweet Classification: Personal Health Mentions using LSTMs

**Course:** CSC4093/DSC4213: Neural Networks and Deep Learning (2024/25)  
**Author:** S20545  

## 📌 Project Overview
This repository contains the code, models, and final report for Programming Assignment 01. The objective of this project is to build sequence models using Recurrent Neural Networks (RNNs) to classify Twitter data (tweets) into two categories:
* **Class 1 (Personal):** Tweets where individuals discuss their own personal health conditions.
* **Class 0 (Non-Personal):** Tweets discussing general health topics or non-personal mentions.

## 🧠 Methodology & Architecture
To tackle the complex, heavily imbalanced natural language dataset, several advanced Deep Learning techniques were implemented using **PyTorch**:

* **Advanced Text Preprocessing:** Custom cleaning pipelines utilizing `emoji.demojize` and `contractions` to standardize messy tweet data, alongside vocabulary reduction to filter rare tokens.
* **Model Architectures:** Built both a **Standard LSTM** and a **Bidirectional LSTM (Bi-LSTM)** with Global Max Pooling and Layer Normalization.
* **Handling Class Imbalance:** Replaced standard Binary Cross-Entropy with a custom **Focal Loss** function, dynamically weighted using calculated positive class multipliers to focus on the minority class.
* **Hyperparameter Optimization:** Leveraged **Optuna** with 5-Fold Stratified Cross-Validation to test 25 different architectural and learning rate combinations to find the optimal global parameters.
* **Hardware Acceleration:** Implemented PyTorch Automatic Mixed Precision (AMP) via `autocast` and `GradScaler` for highly optimized, memory-efficient GPU training.

## 📊 Results
The models were evaluated on an isolated test dataset. The Bi-LSTM outperformed the standard LSTM, specifically in its ability to recall the minority class due to its bidirectional contextual understanding.

| Metric | Standard LSTM | Bi-LSTM |
| :--- | :---: | :---: |
| **Test Accuracy** | 85.05% | **86.07%** |
| **Macro F1-Score** | 0.82 | **0.83** |
| **Recall (Personal - 1)** | 0.76 | **0.79** |

*Confusion matrices and detailed loss/accuracy plots are available in the attached `Report.pdf`.*

## ⚙️ Requirements
To run the Jupyter Notebook/Python scripts locally, you will need the following dependencies:

```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn optuna nltk wordcloud emoji contractions