#!/usr/bin/env python3


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# %% [markdown]
# ### New imports
# Throughout this lab we will introduce new functions and libraries. However,
# we will import them here to emphasize these are the new
# code objects in this lab. Keeping imports near the top
# of a notebook makes the code more readable, since scanning the first few
# lines tells us what libraries are used.

# %%
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import ModelSpec as MS, summarize, poly

# %%
Auto = load_data("Auto")
Auto.columns
# %%
Auto.head()
# %%
Auto_dat = pd.DataFrame(Auto).dropna()
X = pd.DataFrame(
    {"intercept": np.ones(Auto.shape[0]), "horsepower": Auto["horsepower"]}
)
# %%
# model=smf.ols('mpg~horsepower', Auto_dat)
# res=model.fit()
X_test = Auto_dat["horsepower"]
X_test = sm.add_constant(X_test)
Y_test = Auto_dat["mpg"]
model = sm.OLS(Y_test, X_test)
res = model.fit()
Xnew = [1, 98]
predict = res.get_prediction(Xnew)
prediction_summary_frame = predict.summary_frame(alpha=0.05)
confidence_intervals = res.conf_int(alpha=0.05)
print(prediction_summary_frame)
# %%
X = Auto["horsepower"]
Y = Auto["mpg"]
mod = smf.ols("mpg~horsepower", Auto_dat)
results = mod.fit()
res_fit = results.fittedvalues
resid_error = results.resid_pearson
print(results.mse_resid)
# %%
horse = Auto_dat["horsepower"]
mpg = Auto_dat["mpg"]
results.summary()
# %%
design = MS(["horsepower"])
design = design.fit(Auto)
df_98 = pd.DataFrame({"horsepower": [98]})
X_98 = design.transform(df_98)
# %%
import seaborn as sns

ax = plt.subplots(figsize=(8, 8))[1]
# ax.scatter(res.fittedvalues, res.resid)
sns.residplot(
    x="horsepower",
    y="mpg",
    order=2,
    data=Auto_dat,
    lowess=True,
    scatter_kws={"alpha": 0.5},
    line_kws={"color": "red", "lw": 1, "alpha": 0.8},
)
ax.set_xlabel("Fitted value")
ax.set_ylabel("Residual")
ax.axhline(0, c="k", ls="--")
# %%
infl = results.get_influence()
X2 = pd.DataFrame({"intercept": np.ones(Auto.shape[0]), "lstat": Auto["horsepower"]})
sns.scatterplot(x=np.arange(X2.shape[0]), y=infl.hat_matrix_diag)
# %%
g = sns.PairGrid(Auto_dat, diag_sharey=False, height=1, aspect=1)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot, lw=3, legend=False)
pd.plotting.scatter_matrix(Auto_dat)
Auto_dat.corr()
# %%

y = Auto["mpg"]
terms = Auto.columns.drop(["mpg"])
X3 = MS(terms).fit_transform(Auto)
X = Auto_dat["horsepower"]
feat = Auto_dat.drop(columns=["mpg"])
feature_string = " + ".join(terms)
## Fit the model
res2 = smf.ols("mpg ~ " + feature_string, data=Auto_dat).fit()
print(res2.summary())
# %%
import OLSplots

res2fit = res2.fittedvalues
sns.residplot(x=res2fit, y="mpg", data=Auto_dat, lowess=True)
OLSplots.Leverage(res2)
# %%
mod3 = smf.ols(
    "mpg~displacement*horsepower+year*origin+origin*weight+horsepower*year+displacement*weight+horsepower*origin+horsepower*weight+displacement*origin",
    data=Auto_dat,
).fit()
mod3.summary()
# poly = PolynomialFeatures(interaction_only=True,include_bias = False)
# %%

np.random.seed(1)  # In order to generate the same results whenever we run the code
rng = np.random.default_rng(1)
x = rng.normal(size=100)
y = 2 * x + rng.normal(size=100)
df = pd.DataFrame({"x": x, "y": y})
fig, ax = plt.subplots()
sns.regplot(x="x", y="y", data=df, scatter_kws={"s": 50, "alpha": 1}, ax=ax)
ax.axhline(color="gray")
ax.axvline(color="gray")
reg = smf.ols("y~x", df).fit()
reg.summary()
