
# =============================================================
# IRIS FLOWER CLASSIFICATION
# Author: Sanjai KV | GitHub: sanjaikv2255
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

# ── Styling ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f9f9f9",
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
COLORS = {"Iris-setosa": "#4C72B0", "Iris-versicolor": "#DD8452", "Iris-virginica": "#55A868"}

# ── 1. Load & Explore ─────────────────────────────────────────
print("=" * 55)
print("  IRIS FLOWER CLASSIFICATION — CodeAlpha Internship")
print("=" * 55)

df = pd.read_csv("Iris.csv")
df.drop("Id", axis=1, inplace=True)

print("\n📌 Dataset Shape:", df.shape)
print("\n📌 First 5 rows:")
print(df.head())
print("\n📌 Class Distribution:")
print(df["Species"].value_counts())
print("\n📌 Statistical Summary:")
print(df.describe().round(2))
print("\n📌 Missing values:", df.isnull().sum().sum())

# ── 2. EDA Plots ──────────────────────────────────────────────

# Plot 1 — Pairplot
print("\n⏳ Generating plots...")
fig, axes = plt.subplots(4, 4, figsize=(13, 12))
features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
labels = df["Species"].unique()

for i, fx in enumerate(features):
    for j, fy in enumerate(features):
        ax = axes[i][j]
        if i == j:
            for sp in labels:
                subset = df[df["Species"] == sp][fx]
                ax.hist(subset, bins=15, alpha=0.65, color=COLORS[sp], edgecolor="white")
        else:
            for sp in labels:
                sub = df[df["Species"] == sp]
                ax.scatter(sub[fy], sub[fx], alpha=0.7, s=25,
                           color=COLORS[sp], edgecolors="white", linewidths=0.4)
        if i == 3: ax.set_xlabel(fy.replace("Cm", ""), fontsize=8)
        if j == 0: ax.set_ylabel(fx.replace("Cm", ""), fontsize=8)
        ax.tick_params(labelsize=7)

patches = [mpatches.Patch(color=COLORS[s], label=s.replace("Iris-", "")) for s in labels]
fig.legend(handles=patches, loc="upper center", ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, 1.01), frameon=False)
fig.suptitle("Pairplot — Iris Features by Species", fontsize=14, fontweight="bold", y=1.04)
plt.tight_layout()
plt.savefig("plots/01_pairplot.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 2 — Correlation Heatmap
fig, ax = plt.subplots(figsize=(7, 5))
corr = df.drop("Species", axis=1).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            mask=mask, ax=ax, linewidths=0.5,
            cbar_kws={"shrink": 0.8}, annot_kws={"size": 11})
ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("plots/02_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 3 — Boxplots
fig, axes = plt.subplots(1, 4, figsize=(14, 5))
for ax, feat in zip(axes, features):
    data = [df[df["Species"] == sp][feat].values for sp in labels]
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))
    for patch, sp in zip(bp["boxes"], labels):
        patch.set_facecolor(COLORS[sp])
        patch.set_alpha(0.75)
    ax.set_xticklabels([s.replace("Iris-", "") for s in labels], fontsize=8.5)
    ax.set_title(feat.replace("Cm", " (cm)"), fontsize=10, fontweight="bold")
fig.suptitle("Feature Distribution by Species", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/03_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 3. Preprocessing ──────────────────────────────────────────
X = df.drop("Species", axis=1)
le = LabelEncoder()
y = le.fit_transform(df["Species"])  # setosa=0, versicolor=1, virginica=2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n📌 Train size: {len(X_train)} | Test size: {len(X_test)}")

# ── 4. Train Models ───────────────────────────────────────────
models = {
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree":       DecisionTreeClassifier(max_depth=4, random_state=42),
    "Support Vector Machine": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
}

results = {}
print("\n" + "=" * 55)
print("  MODEL EVALUATION RESULTS")
print("=" * 55)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    results[name] = {"model": model, "acc": acc, "y_pred": y_pred}
    print(f"\n{'─'*45}")
    print(f"  {name}")
    print(f"{'─'*45}")
    print(f"  Accuracy: {acc:.2f}%")
    print(classification_report(y_test, y_pred,
          target_names=le.classes_, zero_division=0))

# ── 5. Visualize Results ──────────────────────────────────────

# Plot 4 — Accuracy comparison
fig, ax = plt.subplots(figsize=(8, 5))
names = list(results.keys())
accs = [results[n]["acc"] for n in names]
bar_colors = ["#4C72B0", "#DD8452", "#55A868"]
bars = ax.barh(names, accs, color=bar_colors, height=0.5, edgecolor="white")
for bar, acc in zip(bars, accs):
    ax.text(bar.get_width() - 1.5, bar.get_y() + bar.get_height()/2,
            f"{acc:.1f}%", va="center", ha="right", fontsize=11,
            color="white", fontweight="bold")
ax.set_xlim(85, 102)
ax.set_xlabel("Accuracy (%)", fontsize=11)
ax.set_title("Model Accuracy Comparison", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("plots/04_accuracy_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 5 — Confusion matrices (3 models side by side)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
class_names = [c.replace("Iris-", "") for c in le.classes_]

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, cbar=False, annot_kws={"size": 13})
    ax.set_title(f"{name}\n({res['acc']:.1f}%)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)

fig.suptitle("Confusion Matrices — All Models", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/05_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 6 — Decision boundary (2 best features: PetalLength vs PetalWidth)
best_model_name = max(results, key=lambda k: results[k]["acc"])
best_model = results[best_model_name]["model"]

# Retrain best model on 2 features for visualization
X2 = df[["PetalLengthCm", "PetalWidthCm"]].values
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y, test_size=0.2, random_state=42, stratify=y)
best_model_2f = type(best_model)(**{k: v for k, v in best_model.get_params().items()})
best_model_2f.fit(X2_train, y2_train)

h = 0.02
x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = best_model_2f.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(8, 6))
cmap_bg = plt.cm.get_cmap("Pastel1", 3)
ax.contourf(xx, yy, Z, alpha=0.35, cmap=cmap_bg)
for idx, sp in enumerate(labels):
    mask = df["Species"] == sp
    ax.scatter(df[mask]["PetalLengthCm"], df[mask]["PetalWidthCm"],
               label=sp.replace("Iris-", ""), color=COLORS[sp],
               edgecolors="white", linewidths=0.5, s=55, alpha=0.9)
ax.set_xlabel("Petal Length (cm)", fontsize=11)
ax.set_ylabel("Petal Width (cm)", fontsize=11)
ax.set_title(f"Decision Boundary — {best_model_name}", fontsize=12, fontweight="bold")
ax.legend(title="Species", fontsize=9)
plt.tight_layout()
plt.savefig("plots/06_decision_boundary.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n✅ All 6 plots saved to /plots/")
print("\n🏆 Best Model:", best_model_name, f"({results[best_model_name]['acc']:.2f}%)")
print("\n✅ Project complete! Ready to push to GitHub.\n")
