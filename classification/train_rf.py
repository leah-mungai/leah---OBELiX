from parse import read_xy

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay


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

if cif_only:
    m = train_xy[train_xy["CIF"] == "Match"]
    cm = train_xy[train_xy["CIF"] == "Close Match"]
    train_xy = pd.concat([m, cm], axis=0)
train_xy = train_xy.drop("CIF", axis=1)

x_train = train_xy.iloc[:, :-1].to_numpy()
y_train = train_xy.iloc[:, -1].to_numpy()

idx = np.arange(len(x_train))
np.random.shuffle(idx)
x_train = np.array(x_train)[idx]
y_train = np.array(y_train)[idx]
y_class_train = (y_train > -4)

#conditions = [
#    (y_train > -9) & (y_train <= -6),
#    (y_train > -6) & (y_train <= -3),
#    (y_train > -3)
#]

#labels = [0, 1, 2]

#y_class_train = np.select(conditions, labels)

hparams = {
    "n_estimators": [100],
    "max_depth": [64],
    "max_features": ["sqrt"],
    "min_samples_leaf": [1],
}

hparams_ex = {
    "max_depth": 12,
    "max_features": "sqrt",
    "min_samples_leaf": 2,
    "n_estimators": 50,
}

model = RandomForestClassifier(**hparams_ex)

gs = GridSearchCV(
    estimator=model, param_grid=hparams, cv=5, scoring="neg_mean_absolute_error"
)
gs.fit(x_train, y_class_train)


print("Best parameters:", gs.best_params_)
print(
    "Best RF result:",
    abs(gs.cv_results_["mean_test_score"][gs.best_index_]),
    "±",
    gs.cv_results_["std_test_score"][gs.best_index_],
)

y_scores = gs.best_estimator_.predict_proba(x_train)[:, 1]

custom_threshold = 0.5
y_pred = (y_scores >= 0.5)

accuracy_train = sum((y_pred == y_class_train))/len(y_train)

print(f"Training Classification Accuracy: {accuracy_train:.4f}")

confusion_matrix = confusion_matrix(y_class_train, y_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])
cm_display.plot()

RocCurveDisplay.from_predictions(y_class_train, y_scores)
plt.show()




# Drop CIF == True
# best_params = {
#     "max_depth": 64,
#     "max_features": None,
#     "min_samples_leaf": 1,
#     "n_estimators": 200,
# }

# Drop CIF == False
# best_params = {
#     "max_depth": 36,
#     "max_features": "sqrt",
#     "min_samples_leaf": 1,
#     "n_estimators": 50,
# }

# Partial == False
# best_params = {
#     "max_depth": 36,
#     "max_features": "sqrt",
#     "min_samples_leaf": 1,
#     "n_estimators": 100,
# }

# model = RandomForestRegressor(**best_params, oob_score="neg_mean_absolute_error")
# model.fit(x_train, y_train)
# plt.plot(model.loss_curve_)

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
