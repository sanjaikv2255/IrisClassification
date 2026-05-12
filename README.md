# 🌸 Iris Flower Classification

** Data Science Project 1 **

A machine learning project to classify Iris flowers into three species — *Setosa*, *Versicolor*, and *Virginica* — based on sepal and petal measurements.

---

## 📊 Dataset

- **Source:** UCI Machine Learning Repository
- **Samples:** 150 (50 per class)
- **Features:** SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
- **Target:** Species (3 classes)
- **Missing values:** None

---

## 🔍 Exploratory Data Analysis

| Insight | Finding |
|--------|---------|
| Class balance | Perfectly balanced — 50 samples each |
| Best features | Petal Length & Petal Width (r = 0.96) |
| Easiest class | Iris-setosa is linearly separable |
| Hardest pair | Versicolor vs Virginica overlap slightly |

---

## 🤖 Models Trained

| Model | Accuracy |
|-------|----------|
| K-Nearest Neighbors (k=5) | **100.00%** |
| Support Vector Machine (RBF) | 96.67% |
| Decision Tree (max_depth=4) | 93.33% |

---

## 📈 Visualizations

| Plot | Description |
|------|-------------|
| `01_pairplot.png` | Feature pairplot colored by species |
| `02_heatmap.png` | Feature correlation heatmap |
| `03_boxplots.png` | Feature distributions per species |
| `04_accuracy_comparison.png` | Model accuracy comparison |
| `05_confusion_matrices.png` | Confusion matrix for all 3 models |
| `06_decision_boundary.png` | Decision boundary (Petal features) |

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/sanjaikv2255/IrisClassification
cd IrisClassification

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Run the notebook
jupyter notebook iris_classification.ipynb

# Or run the Python script
python iris_classification.py
```

---

## 🛠 Tech Stack

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Scikit-learn`

---

## 📌 Key Conclusions

- **KNN** achieved perfect 100% accuracy with k=5
- **Petal dimensions** are far more discriminative than sepal dimensions
- **Iris-setosa** is completely linearly separable from the other two species
- All models scored **93%+**, confirming the dataset is well-structured for classification

---

*Made by [Sanjai KV](https://github.com/sanjaikv2255) 
