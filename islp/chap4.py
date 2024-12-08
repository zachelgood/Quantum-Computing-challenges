#!/usr/bin/env python3


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from ISLP import load_data, confusion_table
from ISLP.models import ModelSpec as MS, summarize, poly
import seaborn as sns
import sklearn as sk
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
)  # linear discriminant analysis
from sklearn.discriminant_analysis import (
    QuadraticDiscriminantAnalysis,
)  # quadratic discriminant analysis
from sklearn.neighbors import KNeighborsClassifier  # K nearest neighbours (KNN)
from sklearn.naive_bayes import GaussianNB

# %%
Weekly = load_data("Weekly")
Weekly_dat = pd.DataFrame(Weekly).dropna()
Weekly_dat.describe()
# %%
sns.pairplot(Weekly)
# %%
# %%
predvar = Weekly.columns.drop(["Today", "Year", "Direction"])
lr = sk.linear_model.LogisticRegression()
mod = lr.fit(Weekly_dat[predvar], Weekly_dat["Direction"])
y = Weekly.Direction == "Up"
x = sm.add_constant(Weekly.iloc[:, 1:7])
res = sm.GLM(y, x, family=sm.families.Binomial()).fit()
pred = res.predict()

glm0 = sm.Logit(y, x)
glm0.fit().summary()
gfit = glm0.fit()
probs = gfit.predict()
labels = np.array(["Down"] * 1089)
labels[probs > 0.5] = "Up"
confusion_table(labels, Weekly.Direction)
# %%
y = Weekly["Direction"]
x = Weekly.iloc[:, 1:7]
test = LogisticRegression().fit(x, y)
test.predict(x)
accuracy_score(y, test.predict(x))
confusion_matrix(y, test.predict(x))
conf_mat = confusion_matrix(y, test.predict(x))
ConfusionMatrixDisplay(conf_mat).plot()
# %%
train = Weekly[Weekly["Year"] < 2009]
test = Weekly[(Weekly["Year"] >= 2009)]
X_train = train.Lag2
X_t = np.array(X_train)
X_t = X_t.reshape(len(X_train), 1)
X_t.shape
X_test = test.Lag2
X_ts = np.array(X_test)
X_ts = X_ts.reshape(len(X_test), 1)
Y_train = train.loc[:, "Direction"]
Y_test = test.loc[:, "Direction"]
x_pred = lr.fit(X_t, Y_train).predict(X_ts)
conf_mat = confusion_matrix(Y_test, x_pred)
ConfusionMatrixDisplay(conf_mat).plot()
accuracy_score(Y_test, x_pred)


# %% 13e
lda = LinearDiscriminantAnalysis()
lda_pred = lda.fit(X_t, Y_train).predict(X_ts)
conf_mat_lda = confusion_matrix(Y_test, lda_pred)
ConfusionMatrixDisplay(conf_mat_lda).plot()
accuracy_score(Y_test, lda_pred)
# %%13f
qda = QuadraticDiscriminantAnalysis()
qda_pred = qda.fit(X_t, Y_train).predict(X_ts)
conf_mat_qda = confusion_matrix(Y_test, qda_pred)
ConfusionMatrixDisplay(conf_mat_qda).plot()
accuracy_score(Y_test, qda_pred)
# %%
knn = KNeighborsClassifier(n_neighbors=4)
knn_pred = knn.fit(X_t, Y_train).predict(X_ts)
conf_mat_knn = confusion_matrix(Y_test, knn_pred)
ConfusionMatrixDisplay(conf_mat_knn).plot()
accuracy_score(Y_test, knn_pred)


# %%
qnb = GaussianNB()
qnb_pred = qnb.fit(X_t, Y_train).predict(X_ts)
conf_mat_qnb = confusion_matrix(Y_test, qnb_pred)
ConfusionMatrixDisplay(conf_mat_qnb).plot()
accuracy_score(Y_test, qnb_pred)
# %%Prob 14
Auto = load_data("Auto")
median = Auto["mpg"].median()
Auto["mpg01"] = np.where(Auto["mpg"] < median, 0, 1)
Auto_dat = pd.DataFrame(Auto)
Auto.corr()
# %%
g = sns.PairGrid(Auto, height=2)
g.map_upper(plt.scatter, s=3)
g.map_diag(plt.hist)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.fig.set_size_inches(12, 12)


# %%
# Problem 15
def Power():
    print(2**3)


def Power2(x, a):
    return x**a


def PlotPower(x, a):
    fig, ax = plt.subplots()
    y = Power2(x, a)
    ax.plot(x, Power2(x, a))


PlotPower(np.arange(1, 11), 3)
