#!/usr/bin/env python3


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.common import random_state
from sklearn.pipeline import Pipeline
import statsmodels.formula.api as smf
import statsmodels.api as sm
from ISLP import load_data, confusion_table
from ISLP.models import ModelSpec as MS, summarize, poly
from ISLP.models import sklearn_sm
import seaborn as sns
import sklearn as sk
import sklearn.linear_model as skl
import sklearn.model_selection as skm
import scipy.stats as st
from scipy.stats import bootstrap
from functools import partial
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    accuracy_score,
    r2_score,
    mean_squared_error,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold

# %%
# 8

rng = np.random.default_rng(0)
x = rng.normal(size=100)
eps = rng.normal(size=100)
b0, b1, b2, b3 = 1, 2, 3, 4
y = b0 + b1 * x + b2 * x**2 + b3 * x**3 + eps
# %%
# 9
College = load_data("College")
College["Private"] = College["Private"].map({"Yes": 1, "No": 0})
X = College.drop(columns=["Private", "Apps"], axis=1)
Y = College["Apps"]
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.5, random_state=0)
xtrain1 = sm.add_constant(xtrain)
xtest1 = sm.add_constant(xtest)
mod = sm.OLS(ytrain, xtrain1).fit()
mod.summary()
lin_reg = LinearRegression()
lin_reg.fit(xtrain, ytrain)
lin_reg.score(xtest, ytest)
pred = lin_reg.predict(xtest)
pred2 = mod.predict(xtest1)
r2_score(ytest, pred)
mean_squared_error(ytest, pred2)
# %%
# c
K = 5
kfold = skm.KFold(K, random_state=0, shuffle=True)
alphas = 10 ** np.linspace(8, -2, 100) / Y.std()
param_grid = {"ridge__alpha": alphas}
scaler = StandardScaler(with_mean=True, with_std=True)
ridge = skl.ElasticNet(alpha=alphas[59], l1_ratio=0)
pipe = Pipeline(steps=[("scaler", scaler), ("ridge", ridge)])
pipe.fit(X, Y)
grid = skm.GridSearchCV(pipe, param_grid, cv=kfold, scoring="neg_mean_squared_error")
grid.fit(xtrain1, ytrain)
grid.best_params_["ridge__alpha"]
