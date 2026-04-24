# ransomware-detection-using-deep-learning
"A comparative study of Standalone vs. Hybrid Deep Learning architectures (CNN-LSTM &amp; RNN-DNN) for high-accuracy ransomware detection. Features 99.1% accuracy performance benchmarks and automated preprocessing pipelines."
# Ransomware Detection Using Hybrid Deep Learning Models

## 🛡️ Project Overview
This repository contains the source code and research data for a comparative study on ransomware detection. The core of this research is to demonstrate how **Hybrid Deep Learning** architectures can bridge the "Efficiency Gap" left by standalone models.

By integrating spatial feature extraction with temporal sequence analysis, our proposed **Hybrid CNN-LSTM** model achieves a superior detection rate, effectively mitigating the risks of zero-day and polymorphic ransomware.

## 📊 Key Performance Results
The models were benchmarked against four standalone architectures over a dataset of malware and benign samples.

| Model | Accuracy | F1-Score | Status |
| :--- | :--- | :--- | :--- |
| **Hybrid CNN-LSTM** | **99.1%** | **0.991** | ✨ **Top Performer** |
| **Combined RNN-DNN** | 98.8% | 0.988 | High Efficiency |
| Standalone CNN | 98.5% | 0.984 | Baseline |
| Standalone DNN | 98.2% | 0.981 | Baseline |
| Standalone LSTM | 98.0% | 0.979 | Baseline |
| Standalone RNN | 97.5% | 0.974 | Baseline |

## 🏗️ Architecture Detail
We implemented two primary hybrid strategies:
1. **CNN-LSTM:** Uses `Conv1D` layers for spatial feature engineering, followed by `LSTM` layers to capture behavioral sequences.
2. **RNN-DNN:** Combines `SimpleRNN` layers for temporal memory with a deep `Dense` block for refined classification.



## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Framework:** TensorFlow / Keras
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Optimization:** Adam Optimizer with Exponential Learning Rate Decay

## 📁 Repository Structure
* `data/`: Placeholder for the feature dataset (CSV format).
* `scripts/`: Python source code for training and evaluation.
* `results/`: Confusion matrices and training history plots.
* `requirements.txt`: List of necessary Python dependencies.

## 📈 Visualizations
### Performance Comparison
The hybrid models show a clear lift in accuracy and a reduction in False Negatives compared to standalone variants.

### Training Dynamics
The Hybrid CNN-LSTM utilizes a 50-epoch cycle to achieve deep convergence, outperforming the 10-epoch standalone benchmarks in stability and loss minimization.

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

## ✉️ Contact
**Wend-wooga Konvelbo** [GitHub](https://github.com/wendkonvelbo) | [LinkedIn](YOUR_LINKEDIN_URL)
