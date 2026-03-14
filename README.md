🌻 **Flowers Classification Competition**

**Goal:** Build a robust deep learning model to classify 5 flower species — Daisy, Dandelion, Rose, Sunflower, and Tulip.
**Primary Metric:** `F1-Macro Score` — ensures balanced accuracy across all classes, penalizing bias toward dominant species.

 📁 **Repository Structure**


```

flowers-competition/

├── .github/workflows/main.yml    # Automated Leaderboard Bot

├── baseline/

│   └── model.py                  # Simple 1-layer CNN starting point

├── evaluation/

│   ├── metrics.py                # F1-Macro & Confusion Matrix scripts

│   └── results.md                # Model performance breakdown

├── leaderboard/

│   ├── README.md                 # Live automated rankings

│   └── update.py                 # Leaderboard table builder

├── submissions/

│   └── submission.csv            # Final test set predictions

├── train.py                      # Main training script (3-layer CNN)

├── requirements.txt              # Dependencies (PyTorch, Scikit-learn, etc.)

└── scores.json                   # Participant scores database

```




🧠 **Model Architecture**


A custom **3-layer CNN** built from scratch in PyTorch — no pre-trained weights or transfer learning.


| Component | Details |

|-----------|---------|

| Feature Extraction | 3× `Conv2d` + `ReLU` + `MaxPool2d` |

| Classifier | Flattened → 2 fully connected layers |

| Loss Function | `CrossEntropyLoss` |

| Optimizer | `Adam` — 10 epochs |




🚀 **Performance**


Achieved an **F1-Macro of 0.5884** on the test set.


- ✅ **Best class:** Sunflower — high color distinctiveness made it the easiest to isolate

- ⚠️ **Hardest pair:** Rose vs. Tulip — similar shape and color profiles caused the most confusion




🛠️ **Reproducing the Results**


```bash

# 1. Install dependencies

pip install -r requirements.txt


# 2. Train the model (auto-detects GPU if available)

python train.py


# 3. Evaluate — generates F1 score + full classification report

python evaluation/metrics.py

```




⚙️ **Automated Leaderboard System**


Powered by a custom **GitHub Actions bot**. Every time `scores.json` is updated, it automatically:

1. Sorts participants by accuracy

2. Formats a new Markdown leaderboard table

3. Pushes the update to `leaderboard/README.md`


No manual updates needed — the rankings stay live and current.




> 📌 **Note on Model Weights:** `checkpoint.pth` was trained on Kaggle (GPU T4 ×2). Due to GitHub's 25MB file limit, weights are hosted in the [Kaggle output folder](#) linked to this project.


, comment je peux faire le changement pour avoir le resultat 

📊 **Live Leaderboard:**

## 🏆 Leaderboard

| Rank | User | F1-Macro | Accuracy |
| :--- | :--- | :--- | :--- |
| 1 | ahmedssow-ship-it | 0.6229 | 0.6407 |


📁 **Repository Structure**


```

flowers-competition/

├── .github/workflows/main.yml    # Automated Leaderboard Bot

├── baseline/

│   └── model.py                  # Simple 1-layer CNN starting point

├── evaluation/

│   ├── metrics.py                # F1-Macro & Confusion Matrix scripts

│   └── results.md                # Model performance breakdown

├── leaderboard/

│   ├── README.md                 # Live automated rankings

│   └── update.py                 # Leaderboard table builder

├── submissions/

│   └── submission.csv            # Final test set predictions

├── train.py                      # Main training script (3-layer CNN)

├── requirements.txt              # Dependencies (PyTorch, Scikit-learn, etc.)

└── scores.json                   # Participant scores database

```




🧠 **Model Architecture**


A custom **3-layer CNN** built from scratch in PyTorch — no pre-trained weights or transfer learning.


| Component | Details |

|-----------|---------|

| Feature Extraction | 3× `Conv2d` + `ReLU` + `MaxPool2d` |

| Classifier | Flattened → 2 fully connected layers |

| Loss Function | `CrossEntropyLoss` |

| Optimizer | `Adam` — 10 epochs |




🚀 **Performance**


Achieved an **F1-Macro of 0.5884** on the test set.


- ✅ **Best class:** Sunflower — high color distinctiveness made it the easiest to isolate

- ⚠️ **Hardest pair:** Rose vs. Tulip — similar shape and color profiles caused the most confusion




🛠️ **Reproducing the Results**


```bash

# 1. Install dependencies

pip install -r requirements.txt


# 2. Train the model (auto-detects GPU if available)

python train.py


# 3. Evaluate — generates F1 score + full classification report

python evaluation/metrics.py

```




⚙️ **Automated Leaderboard System**


Powered by a custom **GitHub Actions bot**. Every time `scores.json` is updated, it automatically:

1. Sorts participants by accuracy

2. Formats a new Markdown leaderboard table

3. Pushes the update to `leaderboard/README.md`


No manual updates needed — the rankings stay live and current.




> 📌 **Note on Model Weights:** `checkpoint.pth` was trained on Kaggle (GPU T4 ×2). Due to GitHub's 25MB file limit, weights are hosted in the [Kaggle output folder](#) linked to this project.


, comment je peux faire le changement pour avoir le resultat 

📊 **Live Leaderboard:**

## 🏆 Leaderboard

| Rank | User | F1-Macro | Accuracy |
| :--- | :--- | :--- | :--- |
| 1 | ahmedssow-ship-it | 0.6229 | 0.6407 |
