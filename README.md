# 🧬 Genetic Algorithm Feature Selection - ML Optimizer
**Genetic Algorithm Feature Selection** is an intelligent feature selection tool that uses evolutionary computation to automatically optimize feature subsets for machine learning models. Built with Python and DEAP, it combines correlation analysis with genetic algorithms to find the best features while reducing multicollinearity.

---

## 🚀 Features

### 🎯 Intelligent Feature Selection
- Automatic feature subset optimization using genetic algorithms
- Correlation-based fitness evaluation with target variable
- Multicollinearity penalty to avoid redundant features
- Support for both numerical and categorical variables

### 🔄 Advanced Preprocessing
- Automatic target encoding for categorical variables
- Missing value handling with intelligent imputation
- Data cleaning with column name standardization
- Train-test split generation for model validation

### ⚡ Evolutionary Optimization
- Tournament selection for robust parent selection
- Two-point crossover for feature combination
- Bit-flip mutation with configurable probability
- Real-time generation progress tracking

---

## 🛠️ Tech Stack
- **Core:** Python 3.7+
- **ML Libraries:** scikit-learn, pandas, numpy
- **Genetic Algorithm:** DEAP (Distributed Evolutionary Algorithms)
- **Data Processing:** pandas DataFrame operations
- **Optimization:** Custom correlation-based fitness function

---

## ⚙️ Setup Instructions

### ✅ Clone the Repository
```bash
git clone https://github.com/Slayer9966/Feature-Selection-using-Genetic-Algorithm.git
Feature-Selection-using-Genetic-Algorithm
```

---

### 🐍 Python Environment Setup
```bash
python -m venv venv
venv\Scripts\activate       # For Windows
# or
source venv/bin/activate    # For macOS/Linux

pip install -r requirements.txt

---


```

The algorithm will process your dataset and generate optimized feature subsets along with processed train/test files.

---

## 📊 How to Use

### 📁 Prepare Your Dataset
1. Ensure your dataset is in CSV format
2. Have a clear target/label column
3. Mixed data types (numerical and categorical) are supported

### ⚙️ Configure Parameters
```python
# Update these variables in the script
filepath = 'your_dataset.csv'
target_column = 'your_target_column'

# Customize GA parameters (optional)
selected_features = genetic_feature_selection(
    X_train, y_train, feature_names,
    n_gen=50,        # Number of generations
    pop_size=200,    # Population size
    cxpb=0.7,        # Crossover probability
    mutpb=0.3        # Mutation probability
)
```

### 🎯 Algorithm Components
- **Individual Encoding:** Binary representation (1=selected, 0=excluded)
- **Fitness Function:** `fitness = Σ(correlations) - mean(inter_correlations)`
- **Selection Method:** Tournament selection (tournsize=3)
- **Genetic Operators:** Two-point crossover + bit-flip mutation

---

## 📈 Output Files

### 📊 Generated Files
- `processed_X_train.csv` - Processed training features
- `processed_X_test.csv` - Processed testing features
- `processed_y_train.csv` - Training target values
- `processed_y_test.csv` - Testing target values

### 📋 Console Output
- Generation-by-generation progress tracking
- Average number of features selected per generation
- Final optimized feature subset list

---

## 🔧 Key Functions

### `load_data(filepath, target_column)`
- Loads and preprocesses the dataset
- Handles categorical encoding and missing values
- Returns train-test splits and feature names

### `genetic_feature_selection(X_train, y_train, feature_names, ...)`
- Main genetic algorithm implementation
- Returns list of optimally selected feature names

### `target_encode(X, y, column)`
- Encodes categorical variables using target mean encoding
- Handles missing values appropriately

### `evaluate(individual, ...)`
- Custom fitness function for genetic algorithm
- Balances feature relevance with multicollinearity penalty

---

## 📌 Performance Notes
- **Scalability:** Works efficiently with datasets up to 1000 features
- **Computation Time:** Scales with population size × generations × feature count
- **Memory Usage:** Optimized for standard RAM configurations
- **Convergence:** Typically converges within 30-50 generations

---

## 🎯 Use Cases
- **Feature Engineering:** Reduce feature dimensionality before model training
- **Model Performance:** Improve prediction accuracy by removing noise
- **Multicollinearity:** Eliminate highly correlated redundant features
- **Data Science Pipelines:** Integrate into ML preprocessing workflows

---

## 🔮 Future Enhancements
- Support for non-linear fitness functions
- Integration with specific ML model performance metrics
- Parallel processing for faster execution
- GUI interface for non-technical users
- Support for feature importance from tree-based models

---

## 📜 License
Licensed under the [MIT License](LICENSE) — use, modify, and distribute freely.

---

## 🙋‍♂️ Author
**Your Name**  
📍 Your Location  
📧 your.email@example.com  
🔗 [GitHub](https://github.com/Slayer9966) | [LinkedIn](https://www.linkedin.com/posts/faizan-ali-7b4275297_machinelearning-featureselection-geneticalgorithms-activity-7271489152624336897-gvnq?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEfDpTgBZMmz-8LKpOQTMYhhO24GPrIrPTI)

📢 If you find this project helpful for your machine learning workflows or use it in research, please consider giving it a ⭐ or letting me know via email or GitHub issues!
