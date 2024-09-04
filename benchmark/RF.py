from dave.proxies.data import CrystalFeat
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit


DATA_PATH = Path("/home/felixt/GoogleDrive/Mila/ionic-cond/ionic-conductivity/benchmark/nrcc_ionic_conductivity")
print(DATA_PATH)


# Reconverting data to numpy array
x_mean = torch.load(str(DATA_PATH / "x.mean"), weights_only=True)
x_std = torch.load(str(DATA_PATH / "x.std"), weights_only=True)

trainset = CrystalFeat(root=str(DATA_PATH), target="Ionic conductivity (S cm-1)", subset="train", scalex={"mean":x_mean, "std":x_std})
x_train = []
y_train = []

for i, (x, y) in enumerate(trainset):
    a, b, c = x
    # print(a[a != 0])
    # if i == 10:
    #     break
    b = b.unsqueeze(-1)
    x = torch.cat((a, b, c), dim=-1)
    x_train.append(x.numpy())
    y_train.append(np.log10(y.numpy()))
idx = np.arange(len(x_train))
np.random.shuffle(idx)
x_train = np.array(x_train)[idx]
y_train = np.array(y_train)[idx]


hparams = {
    "n_estimators": [50, 100, 150],
    "max_depth": [12, 24, 36],
    "max_features": ["log2", "sqrt"],
    "min_samples_leaf": [2, 3, 4, 5]
}

hparams_ex = {
    'max_depth': 12,
    'max_features': 'sqrt',
    'min_samples_leaf': 2,
    'n_estimators': 50
}

model = RandomForestRegressor(**hparams_ex)
res = cross_validate(model, x_train, y_train, cv=5, scoring="neg_mean_absolute_error")

print("Example RF result:", -np.mean((res["test_score"])), "±", np.std((res["test_score"])))

gs = GridSearchCV(estimator=model, param_grid=hparams, cv=5, scoring="neg_mean_absolute_error")
gs.fit(x_train, y_train)


print("Best parameters:", gs.best_params_)
print("Best RF result:",  abs(gs.cv_results_["mean_test_score"][gs.best_index_]), "±", gs.cv_results_["std_test_score"][gs.best_index_])
y_train = np.array(y_train).reshape(-1, 1)
y_pred = gs.best_estimator_.predict(x_train)

plt.figure()
plt.scatter(y_train, y_pred)

plt.show()


