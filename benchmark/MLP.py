from parse import read_xy

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import GridSearchCV


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
scaler = scaler.fit(train_xy.iloc[:, -7:-2])
train_xy.iloc[:, -7:-2] = scaler.transform(train_xy.iloc[:, -7:-2])
test_xy.iloc[:, -7:-2] = scaler.transform(test_xy.iloc[:, -7:-2])
# both = [train_xy, test_xy]
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

model = MLPRegressor(**hparams)

# for est in res["estimator"]:
#     loss = est.loss_curve_
#     plt.figure()
#     plt.plot(loss)
#     plt.yscale("log")

#     plt.figure()
#     plt.scatter(y_train, est.predict(x_train))

# plt.savefig("mlp_bm.png")

hparams = {
    "hidden_layer_sizes": [
        [32, 32],
        [16, 16],
        [64, 64],
        [128, 128],
        [256, 256],
        [32, 32, 32],
        [16, 16, 16],
        [64, 64, 64],
        [128, 128, 128],
        [256, 256, 256],
        [32, 32, 32, 32],
        [16, 16, 16, 16],
        [64, 64, 64, 64],
        [128, 128, 128, 128],
        [256, 256, 256, 256],
        [64, 64, 64, 64, 64],
    ],
    "activation": ["relu"],
    "solver": ["adam"],
    "learning_rate_init": [0.003, 0.01, 0.03, 0.1, 0.3],
    "early_stopping": [True],
    "batch_size": [16, 32, 64],
    "max_iter": [1000],
    "learning_rate": ["adaptive"],
    "n_iter_no_change": [100],
}

gs = GridSearchCV(
    estimator=model, param_grid=hparams, cv=5, scoring="neg_mean_absolute_error"
)
gs.fit(x_train, y_train)
# model.fit(x_train[-86:], y_train[-86:])
# model.score(x_train[-86:], y_train[-86:])

# print(gs.best_estimator_)
print("Best parameters:", gs.best_params_)
print(
    "Best MLP result:",
    abs(gs.cv_results_["mean_test_score"][gs.best_index_]),
    "±",
    gs.cv_results_["std_test_score"][gs.best_index_],
)
plt.plot(gs.best_estimator_.loss_curve_)
plt.plot(gs.best_estimator_.validation_scores_)
plt.savefig("mlp_bm.png")
