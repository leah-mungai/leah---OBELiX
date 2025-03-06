from parse import read_xy

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay,roc_curve, roc_auc_score, RocCurveDisplay


np.random.seed(0)

BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "data"
MODEL_PATH = BASE_PATH / "benchmark"

cif_only = False
xy = read_xy(DATA_PATH / "processed.csv")  # , partial=False)

train_idx = [l.strip() for l in open(BASE_PATH / "train_idx.csv")][1:]
test_idx = [l.strip() for l in open(BASE_PATH / "test_idx.csv")][1:]

scaler = StandardScaler()
xy["Ionic conductivity (S cm-1)"] = xy["Ionic conductivity (S cm-1)"].map(np.log10)

train_xy = xy.loc[train_idx]
test_xy = xy.loc[test_idx]
scaler = scaler.fit(train_xy.iloc[:, -8:-2])
train_xy.iloc[:, -8:-2] = scaler.transform(train_xy.iloc[:, -8:-2])
test_xy.iloc[:, -8:-2] = scaler.transform(test_xy.iloc[:, -8:-2])
# both = [train_xy, test_xy]
if cif_only:
    m = train_xy[test_xy["CIF"] == "Match"]
    cm = train_xy[test_xy["CIF"] == "Close Match"]
    test_xy = pd.concat([m, cm], axis=0)
test_xy = test_xy.drop("CIF", axis=1)

x_test = test_xy.iloc[:, :-1].to_numpy()
y_test = test_xy.iloc[:, -1].to_numpy()

idx = np.arange(len(x_test))
np.random.shuffle(idx)
x_test = np.array(x_test)[idx]
y_test = np.array(y_test)[idx]
y_class_test = (y_test > -4)

#conditions = [ 
#    (y_train > -9) & (y_train <= -6),                 
#    (y_train > -6) & (y_train <= -3),
#    (y_train > -3)
#]

#labels = [0, 1, 2]

#y_class_train = np.select(conditions, labels)

hparams = {
    "hidden_layer_sizes": [32, 32],
    "activation": "relu",
    "solver": "adam",
    "learning_rate_init": 0.01,
    "early_stopping": True,
    "batch_size": 32,
    "max_iter": 1000,
    "learning_rate": "adaptive",
    "n_iter_no_change": 100,
}

model = MLPClassifier(**hparams)

hparams = {
    "hidden_layer_sizes": [
        [32, 32],
 # [16, 16],
 # [64, 64],
 # [128, 128],
 # [256, 256],
 # [32, 32, 32],
 # [16, 16, 16],
 # [64, 64, 64],
 # [128, 128, 128],
 # [256, 256, 256],
 # [32, 32, 32, 32],
 # [16, 16, 16, 16],
 # [64, 64, 64, 64],
 # [128, 128, 128, 128],
 # [256, 256, 256, 256],
 # [64, 64, 64, 64, 64],
    ],
    "activation": ["relu"],
    "solver": ["adam"],
    "learning_rate_init": [0.003],#, 0.01, 0.03, 0.1, 0.3],
    "early_stopping": [True],
    "batch_size": [16, 32],#, 64],
    "max_iter": [1000],
    "learning_rate": ["adaptive"],
    "n_iter_no_change": [100],
}

gs = GridSearchCV(
    estimator=model, param_grid=hparams, cv=5, scoring="neg_mean_absolute_error"
)

gs.fit(x_test, y_class_test)

print("Best parameters:", gs.best_params_)
print(
    "Best MLP result:",
    abs(gs.cv_results_["mean_test_score"][gs.best_index_]),
    "±",
    gs.cv_results_["std_test_score"][gs.best_index_],

 )

y_scores = gs.best_estimator_.predict_proba(x_test)[:, 1]

custom_threshold = 0.5 
y_pred = (y_scores >= 0.5)

accuracy_test = sum((y_pred == y_class_test))/len(y_test)

print(f"Testing Classification Accuracy: {accuracy_test:.4f}")

confusion_matrix =confusion_matrix(y_class_test, y_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])
cm_display.plot()

RocCurveDisplay.from_predictions(y_class_test, y_scores)
plt.show()

#fpr, tpr, thresholds = roc_curve(y_class_train, y_scores)
#auc_score = roc_auc_score(y_class_train, y_scores)
#print(f"AUC Score: {auc_score:.4f}")

#plt.figure(figsize=(8, 6))
#plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
#plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Random classifier line
#plt.xlabel("False Positive Rate (FPR)")
#plt.ylabel("True Positive Rate (TPR)")
#plt.title("ROC Curve")
#plt.legend()
#plt.show()

# plt.plot(gs.best_estimator_.loss_curve_)
# plt.plot(gs.best_estimator_.validation_scores_)
# plt.yscale("log")
# plt.savefig("mlp_bm.png")

# Drop CIF = True
# best_params = {
#     "activation": "relu",
#     "batch_size": 16,
#     "early_stopping": True,
#     "hidden_layer_sizes": [16, 16, 16],
#     "learning_rate": "adaptive",
#     "learning_rate_init": 0.01,
#     "max_iter": 1000,
#     "n_iter_no_change": 100,
#     "solver": "adam",
# }


# Drop CIF = False
# best_params = {
#     "activation": "relu",
#     "batch_size": 16,
#     "early_stopping": True,
#     "hidden_layer_sizes": [64, 64, 64, 64],
#     "learning_rate": "adaptive",
#     "learning_rate_init": 0.003,
#     "max_iter": 1000,
#     "n_iter_no_change": 100,
#     "solver": "adam",
# }

# Partial = False
# best_params = {
#     "activation": "relu",
#     "batch_size": 16,
#     "early_stopping": True,
#     "hidden_layer_sizes": [64, 64, 64, 64, 64],
#     "learning_rate": "adaptive",
#     "learning_rate_init": 0.01,
#     "max_iter": 1000,
#     "n_iter_no_change": 100,
#     "solver": "adam",
# }

# model = MLPRegressor(**best_params)
# model.fit(x_train, y_train)

# for cif_only in [True, False]:
#     if cif_only:
#         m = test_xy[test_xy["CIF"] == "Match"]
#         cm = test_xy[test_xy["CIF"] == "Close Match"]
#         test = pd.concat([m, cm], axis=0)
#         test = test.drop("CIF", axis=1)
#     else:
#         test = test_xy.drop("CIF", axis=1)

#     x_test = test.iloc[:, :-1].to_numpy()
#     y_test = test.iloc[:, -1].to_numpy()
#     y_pred = model.predict(x_test)
#     loss = mean_absolute_error(y_test, y_pred)
#     if cif_only:
#         print("CIF Only", loss)
#     else:
#         print("Whole dataset", loss)
