

---

# 🧬 GeneticFS Engine Pro - Hybrid ML Optimizer

**GeneticFS Engine Pro** is an advanced feature selection framework that uses evolutionary computation to prune high-dimensional datasets. Unlike standard selectors, this engine uses a **Hybrid Fitness Function** that combines non-linear statistical analysis with model-based importance to eliminate noise and redundancy.

---

## 🚀 Features

### 🎯 Hybrid Relevance Scoring

* **Spearman Rank Correlation:** Captures non-linear rank relationships that standard Pearson correlation misses.
* **Random Forest Importance:** Uses model-driven Gini impurity (Classification) or MSE (Regression) to find true predictive power.
* **Weighted Integration:** Uses a 60/40 weighted split favoring Random Forest for smarter, more reliable selection.

### 🛡️ Ruthless Pruning Logic

* **Pre-flight Variance Filter:** Automatically kills "Zero-Variance" columns (like constant junk) before the GA begins.
* **Squared Redundancy Penalty:** Heavily punishes multicollinearity by squaring inter-feature correlation, forcing a diverse subset.
* **Aggressive Sparsity Pressure:** Enforces a high cost-per-feature (0.15) to ensure only "heavy-hitters" survive.

### 🔄 Industrial Preprocessing

* **Automated Label Encoding:** Automatically converts categorical string targets (Yes/No, True/False) into binary integers.
* **Target Encoding:** Handles high-cardinality categorical features using target-mean mapping.
* **FastAPI Integration:** Asynchronous job processing with real-time generation tracking and progress monitoring.

---

## 🛠️ Tech Stack

* **Core:** Python 3.9+
* **API Framework:** FastAPI & Uvicorn
* **Genetic Algorithm:** DEAP (Distributed Evolutionary Algorithms)
* **ML Libraries:** scikit-learn (Random Forest, LabelEncoder)
* **Data Processing:** pandas, numpy

---

## ⚙️ Setup Instructions

### ✅ Clone the Repository

```bash
git clone https://github.com/Slayer9966/Feature-Selection-using-Genetic-Algorithm.git
cd Feature-Selection-using-Genetic-Algorithm

```

### 🐍 Python Environment Setup

```bash


pip install -r requirements.txt

```

### ▶️ Run the Server

```bash
python server.py

```

*Access the web interface at `http://127.0.0.1:8000`.*

---

## 📊 Algorithm Mechanics

### 🧬 Evolutionary Configuration

* **Individual Encoding:** Binary representation (1 = Selected, 0 = Excluded)
* **Selection Method:** Tournament selection (tournsize=7) for high selection pressure.
* **Genetic Operators:** Two-point crossover and low-probability (0.02) bit-flip mutation to preserve elite sets.

### ⚖️ The Fitness Formula

The engine evaluates individuals based on a multi-objective weighted formula:


$$Fitness = (0.6 \times RF\_Score + 0.4 \times Spearman) - (Redundancy^2 \times 8.0) - (0.15 \times N\_Features)$$

---

## 📈 Key Improvements (v2.0)

| Component | Old Version | Pro Version |
| --- | --- | --- |
| **Statistical Method** | Pearson Correlation | **Spearman Rank Correlation** |
| **Model Awareness** | None | **Random Forest Importance** |
| **Redundancy** | Linear Subtraction | **Squared Heavy Penalty (8.0x)** |
| **Junk Handling** | Passive | **Pre-flight Variance Filtering** |
| **Target Support** | Numeric Only | **Automated Label Encoding** |

---

## 🎯 Use Cases

* **Noise Reduction:** Stripping out random columns in high-dimensional datasets.
* **Dimensionality Reduction:** Reducing model complexity for edge device deployment.
* **Multicollinearity:** Eliminating highly correlated redundant features in housing or financial data.
* **Discovery:** Finding non-obvious predictors using the hybrid importance engine.

---

## 📜 License

Licensed under the [MIT License](https://www.google.com/search?q=LICENSE) — use, modify, and distribute freely.

---

## 🙋‍♂️ Author

**Syed Muhammad Faizan Ali** 📍 Islamabad, Pakistan

📧 your.email@example.com

🔗 [GitHub](https://github.com/Slayer9966) | [LinkedIn](https://www.google.com/search?q=https://www.linkedin.com/posts/faizan-ali-7b4275297_machinelearning-featureselection-geneticalgorithms-activity-7271489152624336897-gvnq)



---

