from parse import read_xy

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from pathlib import Path

# from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit


BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "data"
MODEL_PATH = BASE_PATH / "benchmark"

cif = True
xy = read_xy(DATA_PATH / "processed.csv")

train_idx = [l.strip() for l in open(BASE_PATH / "train_idx.csv")][1:]
test_idx = [l.strip() for l in open(BASE_PATH / "test_idx.csv")][1:]

scaler = StandardScaler()
xy["Ionic conductivity (S cm-1)"] = xy["Ionic conductivity (S cm-1)"].map(np.log10)

train_xy = xy.loc[train_idx]
test_xy = xy.loc[test_idx]
scaler = scaler.fit(train_xy.iloc[:, -7:-1])
train_xy.iloc[:, -7:-1] = scaler.transform(train_xy.iloc[:, -7:-1])
test_xy.iloc[:, -7:-1] = scaler.transform(test_xy.iloc[:, -7:-1])
both = [train_xy, test_xy]
if cif:
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


hparams = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [12, 24, 36, 48, 56, 64],
    "max_features": ["log2", "sqrt"],
    "min_samples_leaf": [1, 2, 3, 4, 5],
}

hparams_ex = {
    "max_depth": 12,
    "max_features": "sqrt",
    "min_samples_leaf": 2,
    "n_estimators": 50,
}

model = RandomForestRegressor(**hparams_ex)
# res = cross_validate(model, x_train, y_train, cv=5, scoring="neg_mean_absolute_error")

# print(
#     "Example RF result:",
#     -np.mean((res["test_score"])),
#     "±",
#     np.std((res["test_score"])),
# )

gs = GridSearchCV(
    estimator=model, param_grid=hparams, cv=5, scoring="neg_mean_absolute_error"
)
gs.fit(x_train, y_train)


print("Best parameters:", gs.best_params_)
print(
    "Best RF result:",
    abs(gs.cv_results_["mean_test_score"][gs.best_index_]),
    "±",
    gs.cv_results_["std_test_score"][gs.best_index_],
)
# y_train = np.array(y_train).reshape(-1, 1)
# y_pred = gs.best_estimator_.predict(x_train)

# plt.figure()
# plt.scatter(y_train, y_pred)

# plt.show()
