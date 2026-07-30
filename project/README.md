# ☠️ Dead Men Tell No Tales  
### ☠️🍄 Poisonous Mushroom Classification

> *“Dead men tell no tales… but mushrooms do — if you know how to listen.”*


Classic binary classification project that predicts whether a mushroom is **edible** or **poisonous** using the UCI Agaricus-Lepiota dataset.

**Original version** of this project was implemented in **WEKA**.  
This repository contains a modern re-implementation in **Python + scikit-learn**.

Part of the **PUCIT Machine Learning Lectures** series (Nazar Khan).

---

# 🍄 Mushroom Classification – Classical Machine Learning Benchmark

## Project Goal

- Reproduce and modernize the classic “Dead Men Tell No Tales” poisonous mushroom classification project
- Compare **8 classical machine learning algorithms**
- Use proper evaluation: stratified split + **5-fold cross-validation** + held-out test set
- Identify the best model and analyze its predictions (especially false negatives)
- Bridge the old WEKA workflow to the modern Python data-science stack

---

## Models Compared

| Family                | Model                        |
|-----------------------|------------------------------|
| Linear                | Logistic Regression          |
| Tree                  | Decision Tree                |
| Ensemble (Bagging)    | Random Forest                |
| Ensemble (Boosting)   | Gradient Boosting            |
| Kernel                | SVM (RBF)                    |
| Instance-based        | K-Nearest Neighbors          |
| Probabilistic         | Gaussian Naive Bayes         |
| Neural Network        | Multi-Layer Perceptron (MLP) |

---

## Dataset

- **Source**: [UCI Machine Learning Repository – Mushroom](https://archive.ics.uci.edu/ml/datasets/mushroom)
- **Samples**: 8,124
- **Features**: 22 categorical attributes
- **Target**: Binary (`edible` / `poisonous`)
- **Class balance**: Nearly balanced (~52% edible, ~48% poisonous)

---

## Original vs Modern Version

| Aspect              | Original (WEKA)              | Modern Version (this repo)          |
|---------------------|------------------------------|-------------------------------------|
| Tool                | WEKA                         | Python + scikit-learn               |
| Interface           | GUI / Explorer               | Jupyter Notebook                    |
| Evaluation          | Mostly train/test or CV      | 5-fold CV + stratified hold-out     |
| Models              | Selected WEKA classifiers    | 8 carefully chosen classical models |
| Reproducibility     | Manual                       | Fully scripted & reproducible       |
| Visualization       | WEKA plots                   | Seaborn + Matplotlib                |

---

## How to Run

1. Open the notebook in Jupyter / VS Code / Google Colab
2. Run all cells in order
3. The notebook will:
   - Download the dataset automatically
   - Encode categorical features
   - Train all models with 5-fold CV
   - Show a comparison table + bar plot
   - Generate a detailed report for the best model

**Requirements**:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Project Structure

```
├── Dead_Men_Tell_No_Tales.ipynb          # Main notebook
├── Mushroom_Classification_Report.md     # Full written report
└── README.md
```

---

## Key Takeaways

1. **Categorical data** needs encoding (`LabelEncoder` is the simplest starting point).
2. Always use **stratified split + cross-validation** for reliable comparison.
3. **Ensemble methods** (Random Forest, Gradient Boosting) dominate on this dataset.
4. High accuracy is not enough — in safety-critical tasks, look at the **confusion matrix** and the cost of false negatives.
5. Moving from WEKA to scikit-learn gives full control, better reproducibility, and easier integration with modern tools.

---

## Possible Next Steps

- Try One-Hot Encoding instead of Label Encoding
- Analyze feature importance
- Add cost-sensitive learning or threshold tuning
- Experiment with XGBoost / LightGBM / CatBoost
- Visualize the data with PCA or t-SNE

---

## License & Credits

- Dataset: UCI Machine Learning Repository
- Original concept: “Dead Men Tell No Tales” – Poisonous Mushroom Classification (WEKA)
- Modern re-implementation prepared for the PUCIT ML lecture series.
```
