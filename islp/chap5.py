#!/usr/bin/env python3


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.common import random_state
import statsmodels.formula.api as smf
import statsmodels.api as sm
from ISLP import load_data, confusion_table
from ISLP.models import ModelSpec as MS, summarize, poly
from ISLP.models import sklearn_sm
import seaborn as sns
import sklearn as sk
import scipy.stats as st
from scipy.stats import bootstrap
from functools import partial
from sklearn.base import clone
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, KFold


# %%
# Question 5a

np.random.seed(100)
Default = load_data("Default")
# Default["default_yes"] = (Default["default"] == "Yes").astype("int")
Default.head()
mod1 = smf.glm(
    formula="default~income+balance", data=Default, family=sm.families.Binomial()
).fit()
mod1.summary()
# %%

lr = LogisticRegression()
# %%
# b
X = Default[["balance", "income"]]
Y = Default["default"]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)
x_test2 = sm.add_constant(x_test)
x_train2 = sm.add_constant(x_train)
y1_train = []
y1_test = []

for el in y_train:
    if el == "Yes":
        y1_train.append(1)
    elif el == "No":
        y1_train.append(0)

for el in y_test:
    if el == "Yes":
        y1_test.append(1)
    elif el == "No":
        y1_test.append(0)
# %%
# ii-iv
mod_test = lr.fit(x_train, y1_train)
mod_test.coef_
mod2 = sm.GLM(y1_train, x_train2, family=sm.families.Binomial()).fit()
mod2.summary()
pred = mod2.predict(exog=x_test2)
pred.describe()
pred2 = mod_test.predict(x_test)
1 - accuracy_score(y1_test, pred2)
labels = []
for el in pred:
    if el > 0.5:
        labels.append("Yes")
    else:
        labels.append("No")
1 - np.mean(y_test == labels)  # error
# %%
ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=labels)
# %%
# iii


def percent_split(percent):
    if percent < 1:
        pass
    percent = percent / 100
    X = Default[["balance", "income"]]
    Y = Default["default"]
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=percent, random_state=0
    )
    x_test2 = sm.add_constant(x_test)
    x_train2 = sm.add_constant(x_train)

    y1_train = []
    y1_test = []

    for el in y_train:
        if el == "Yes":
            y1_train.append(1)
        elif el == "No":
            y1_train.append(0)

    for el in y_test:
        if el == "Yes":
            y1_test.append(1)
        elif el == "No":
            y1_test.append(0)
    pred = mod2.predict(exog=x_test2)
    labels = []
    for el in pred:
        if el > 0.5:
            labels.append("Yes")
        else:
            labels.append("No")
    return 1 - np.mean(y_test == labels)


# %%

model_set = pd.DataFrame()
model_set["Model"] = None
model_set["Error"] = None
np.random.seed(100)
for i in range(4):
    model_set.loc[len(model_set)] = [
        f"Logistic regression ({round((1-0.2-i/10)*100)}%/{round((0.2+ i/10 )*100)}%)",
        percent_split(20 + i * 10),
    ]
model_set
percent_split(50)
# %%
# d
# %%
# 6
np.random.seed(0)

Default = load_data("Default")
X = Default[["balance", "income"]]
Y = Default["default"]
Y_ = Default.default == "Yes"
# X = sm.add_constant(X)
# glmtest = sm.GLM(Y_, X, family=sm.families.Binomial()).fit()
# glmtest.summary()
mod1 = smf.glm(
    formula="default~income+balance", data=Default, family=sm.families.Binomial()
).fit()
mod1.bse
# %%
# b/c


def boot_fn(D, idx):
    D_ = D.loc[idx]
    Y_ = D_.default == "Yes"
    X_ = D_[["income", "balance"]]
    X_ = sm.add_constant(X_)
    return sm.GLM(Y_, X_, family=sm.families.Binomial()).fit().params


def boot_SE(func, D, n=None, B=100, seed=0):
    rng = np.random.default_rng(seed)
    first_, second_ = 0, 0
    n = n or D.shape[0]
    for _ in range(B):
        idx = rng.choice(D.index, n, replace=True)
        value = func(D, idx)
        first_ += value
        second_ += value**2
    return np.sqrt(second_ / B - (first_ / B) ** 2)


mod1.bse.iloc[1]
res1 = boot_SE(boot_fn, Default, B=100, seed=0)
res1
# %%
#
models_std_errors = pd.DataFrame()
models_std_errors["Model"] = None
models_std_errors["Test_error_income"] = None
models_std_errors["Test_error_balance"] = None
models_std_errors.loc[len(models_std_errors)] = [
    "Logistic regression",
    mod1.bse.iloc[1],
    mod1.bse.iloc[2],
]
models_std_errors.loc[len(models_std_errors)] = [
    "Logistic regression with bootstrap",
    res1.iloc[1],
    res1.iloc[2],
]
pd.set_option("max_colwidth", 400)

models_std_errors
# %%


def testing(func, D, n=None, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    intercept, coeff_balance, coeff_income = [], [], []
    n = None or D.shape[0]
    for _ in range(B):
        idx = rng.choice(D.index, n, replace=True)
        [inter, balance, income] = func(Default, idx)
        intercept.append(inter)
        coeff_balance.append(balance)
        coeff_income.append(income)
    intercept_statistics = {
        "estimated_value": np.mean(intercept),
        "std_error": np.std(intercept),
    }
    balance_statistics = {
        "estimated_value": np.mean(coeff_balance),
        "std_error": np.std(coeff_balance),
    }
    income_statistics = {
        "estimated_value": np.mean(coeff_income),
        "std_error": np.std(coeff_income),
    }
    return {
        "intercept": intercept_statistics,
        "balance_statistices": balance_statistics,
        "income_statistics": income_statistics,
    }


testing(boot_fn, Default, B=1000, seed=0)
# %%
# 7
Weekly = load_data("Weekly")
y = Weekly.Direction == "Up"
Y = Weekly.Direction
x = Weekly[["Lag1", "Lag2"]]
x = sm.add_constant(x)
week_mod = sm.GLM(y, x, family=sm.families.Binomial()).fit()
# %%
# b
x2 = x.drop(0)
y2 = y.drop(0)
week_mod2 = sm.GLM(y2, x2, family=sm.families.Binomial()).fit()
week_mod2.summary()
# %%
# c
x.iloc[0]
probs = week_mod2.predict(x.iloc[0])
round(probs)
Weekly.iloc[0].Direction
prob_comp = [round(probs.iloc[0]), y[0]]
prob_comp
# %%
# d/e
n = len(Weekly)
errors = np.zeros(n)
for i in range(n):
    xtrain = x.drop([i])
    ytrain = y.drop([i])
    week_test = sm.GLM(ytrain, xtrain, family=sm.families.Binomial()).fit()
    prob = week_test.predict(x.iloc[i])
    labels = np.array(["Down"] * 1)
    labels[prob > 0.5] = "Up"
    if labels[0] != Y[i]:
        errors[i] = 1
errors.mean()
# %%
# 8
rng = np.random.default_rng(1)
x = rng.normal(size=100)
y = x - 2 * x**2 + rng.normal(size=100)
# n is number of observations=100, while p is number of predictors: 2 (x and x**2)
plt.scatter(x, y)
# %%c
df = pd.DataFrame({"x": x, "y": y})
X = df["x"].values.reshape(-1, 1)
Y = df["y"].values.reshape(-1, 1)
for i in range(1, 5):
    poly = PolynomialFeatures(i)
    predictors = poly.fit_transform(X)
    lr = LinearRegression()
    # loocv=KFold(100)
    # lm=lr.fit(predictors,Y)
    error = cross_val_score(
        lr, predictors, Y, cv=len(X), scoring="neg_mean_squared_error"
    ).mean()
    print("For model {} , error is {}".format(i, error))
# %%
for i in range(1, 5):
    poly = PolynomialFeatures(i)
    pred = poly.fit_transform(X)
    res = sm.OLS(Y, pred).fit()
    print(res.summary())

Xtest = MS([poly("x", degree=4)], intercept=False).fit_transform(df)
modtest = sm.OLS(Y, Xtest).fit()
modtest.summary()
# %%
# 9
np.random.seed(0)
Boston = load_data("Boston")
mu = Boston["medv"].mean()
SE = Boston["medv"].std() / np.sqrt(len(Boston["medv"]))
SE


# %%
# c
def mean_func(idx):
    return Boston["medv"].iloc[idx].mean()


def boot_SE(func, n=None, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    first_, second_ = 0, 0
    n = n or Boston.shape[0]
    for _ in range(B):
        idx = rng.choice(Boston.index, n, replace=True)
        value = func(idx)
        first_ += value
        second_ += value**2
    return np.sqrt(second_ / B - (first_ / B) ** 2)


(mu - 2 * boot_SE(mean_func, B=1000, seed=0))
# %%

medv = Boston["medv"]
means = []
for _ in range(1000):
    means.append(
        Boston["medv"].sample(n=Boston.shape[0], replace=True, random_state=rng).mean()
    )
boot_std = np.std(means)
boot_std
# %%
medv_s = (medv,)
res = bootstrap(
    medv_s, np.mean, confidence_level=0.9545, axis=0, n_resamples=1000, random_state=0
)
res.standard_error
mean_test = np.mean(res.bootstrap_distribution)
mean_test2 = (res.confidence_interval[0] + res.confidence_interval[1]) / 2
mean_test2 - 2 * res.standard_error
# %%
# d
print(
    "Bootstrap confidence range is",
    res.confidence_interval[0],
    "and",
    res.confidence_interval[1],
)

print("State confidence range is", mu - 2 * SE, "and", mu + 2 * SE)

st.t.interval(0.9545, df=len(Boston) - 1, loc=mean_test2, scale=res.standard_error)
# %%
# e
medv.median()
res2 = bootstrap(
    medv_s,
    np.median,
    confidence_level=0.9545,
    axis=0,
    n_resamples=1000,
    random_state=None,
)


def median_func(idx):
    return Boston["medv"].iloc[idx].median()


boot_SE(median_func, B=1000, seed=0)
res2.standard_error
# %%
# g/h
np.percentile(medv, 10)


def perc_func(idx):
    return np.percentile(Boston["medv"].iloc[idx], 10)


boot_SE(perc_func, B=1000, seed=0)
