from parse import read_xy

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


from sklearn.experimental import enable_halving_search_cv # noqa

from sklearn.model_selection import HalvingGridSearchCV, train_test_split

#from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay,roc_curve, roc_auc_score, RocCurveDisplay


np.random.seed(0)

BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "data"
MODEL_PATH = BASE_PATH / "benchmark"

cif_only = False
xy = read_xy(DATA_PATH / "processed.csv")  # , partial=False)

train_idx = [l.strip() for l in open(DATA_PATH / "train_idx.csv")][1:]
test_idx = [l.strip() for l in open(DATA_PATH / "test_idx.csv")][1:]

scaler = StandardScaler()
xy["Ionic conductivity (S cm-1)"] = xy["Ionic conductivity (S cm-1)"].map(np.log10)

train_xy = xy.loc[train_idx]
test_xy = xy.loc[test_idx]
scaler = scaler.fit(train_xy.iloc[:, -8:-2])
train_xy.iloc[:, -8:-2] = scaler.transform(train_xy.iloc[:, -8:-2])
test_xy.iloc[:, -8:-2] = scaler.transform(test_xy.iloc[:, -8:-2])
# both = [train_xy, test_xy]
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
#y_class_train = (y_train > -6)

conditions = [ 
    (y_train > -9) & (y_train <= -6),                 
    (y_train > -6) & (y_train <= -3),
    (y_train > -3)
]

labels = [0, 1, 2]

y_class_train = np.select(conditions, labels)



if cif_only:
    m_ = test_xy[test_xy["CIF"] == "Match"]
    cm_ = test_xy[test_xy["CIF"] == "Close Match"]
    test_xy = pd.concat([m_, cm_], axis=0)
test_xy = test_xy.drop("CIF", axis=1)

x_test = test_xy.iloc[:, :-1].to_numpy()
y_test = test_xy.iloc[:, -1].to_numpy()

idx_ = np.arange(len(x_test))
np.random.shuffle(idx_)
x_test = np.array(x_test)[idx_]
y_test = np.array(y_test)[idx_]
#y_class_test = (y_test > -6)

conditions = [ 
    (y_test > -9) & (y_test <= -6),                 
    (y_test > -6) & (y_test <= -3),
    (y_test > -3)
]

labels = [0, 1, 2]

y_class_test = np.select(conditions, labels)


model = RandomForestClassifier()

hparams = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [12, 24, 36, 48, 56, 64, None],
    "max_features": ["log2", "sqrt", None],
    "min_samples_leaf": [1, 2, 3, 4, 5],
}


halving_cv = HalvingGridSearchCV(
    estimator=model, param_grid=hparams, factor=3, cv=5, scoring="neg_mean_absolute_error"
)

halving_cv.fit(x_train, y_class_train)

print("Best parameters:", halving_cv.best_params_)
print(
    "Best RF result:",
    abs(halving_cv.cv_results_["mean_test_score"][halving_cv.best_index_]),
    "±",
    halving_cv.cv_results_["std_test_score"][halving_cv.best_index_],

 )

#joblib.dump(gs.best_estimator_, "best_rf_model.pkl")
#
#best_rf = joblib.load("best_rf_model.pkl")
#
#y_pred_train = best_rf.predict(x_train)
#
#y_pred_test = best_rf.predict(x_test)
#
##y_scores = best_rf.predict_proba(x_train)[:, 1]
#
##y_scores_ = best_rf.predict_proba(x_test)[:, 1]
#
##custom_threshold = 0.5
##y_pred = (y_scores >= 0.5)
##y_pred_ = (y_scores_ >= 0.5)
#
#accuracy_train = sum((y_pred_train == y_class_train))/len(y_train)
#
#accuracy_test = sum((y_pred_test == y_class_test))/len(y_test)
#
#print(f"Training Classification Accuracy: {accuracy_train:.4f}", f"Testing Classification Accuracy: {accuracy_test:.4f}")
#
#
##confusion_matrix_1 =confusion_matrix(y_class_train, y_pred_train)
##cm_display_1 = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_1, display_labels = [0, 1, 2])
##cm_display_1.plot()
#
#
#confusion_matrix_2 =confusion_matrix(y_class_test, y_pred_test)
#cm_display_2 = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_2, display_labels = [0, 1, 2])
#cm_display_2.plot()
#
#
#plt.savefig("confusion_matrix_RF.png", dpi=300)
#
#
##RocCurveDisplay.from_predictions(y_class_train, y_scores)
##RocCurveDisplay.from_predictions(y_class_test, y_scores_)
#plt.show()
#
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

