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
    "hidden_layer_sizes": [32, 32],
    "activation": "relu",
    "solver": "adam",
    "learning_rate_init": 0.01,
    # "validation_fraction" :0.2,
    "early_stopping": True,
    "batch_size": 32,
    "max_iter": 1000,
    "learning_rate": "adaptive",
    "n_iter_no_change": 100
}

model = MLPRegressor(**hparams)
res = cross_validate(model, x_train, y_train.reshape((-1,)), cv=5, scoring="neg_mean_absolute_error", return_train_score=True, return_estimator=True)
print("Train scores", res["train_score"])
print(" Test scores", res["test_score"])
print("Example RF result:", -np.mean((res["test_score"])), "±", np.std((res["test_score"])))

for est in res["estimator"]:
    loss = est.loss_curve_
    plt.figure()
    plt.plot(loss)
    plt.yscale("log")

    plt.figure()
    plt.scatter(y_train, est.predict(x_train))

plt.show()

# gs = GridSearchCV(estimator=model, param_grid=hparams, cv=5)
# model.fit(x_train[-86:], y_train[-86:])
# model.score(x_train[-86:], y_train[-86:])
# best = gs.best_score_
# print(best)
# print(gs.best_estimator_)
# print(gs.best_params_)


# In[ ]:




