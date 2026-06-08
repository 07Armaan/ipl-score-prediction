# %%
import pandas as pd
import numpy as np

# %%
print("pandas=="+pd.__version__)
print("numpy=="+np.__version__)

# %%

df = pd.read_csv("t20_powerplay_dataset_250.csv")

# %% [markdown]
# Data Understanding

# %%
df.head()

# %%
df.tail()

# %%
df.shape

# %%
df.duplicated().sum()

# %%
df.isnull().sum()

# %%
df.describe()

# %%
df.info()

# %%
num_cols = df.select_dtypes(include=["int64","float64"]).drop(columns=["final_score"]).columns
cat_cols = df.select_dtypes(include="object").columns
tar_col = df["final_score"]

# %% [markdown]
# EDA

# %%
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# %%
print("matplotlib=="+matplotlib.__version__)
print("seaborn=="+sns.__version__)

# %% [markdown]
# Univariate Analysis

# %% [markdown]
# num cols

# %%
for col in num_cols:
    sns.histplot(x=col,data=df,kde=True)
    plt.title(col)
    plt.show()

# %%
for col in num_cols:
    sns.boxplot(x=col,data=df)
    plt.title(col)
    plt.show()

# %% [markdown]
# cat cols

# %%
for col in cat_cols:
    sns.countplot(x=col,data=df)
    plt.title(col)
    plt.show()

# %% [markdown]
# Target col

# %%
sns.histplot(x=tar_col,data=df,kde=True)
plt.title("Target col")
plt.show()
sns.boxplot(x=tar_col,data=df)
plt.title("Target col")
plt.show()

# %% [markdown]
# Bivariate Analysis

# %%
for col in num_cols:
    sns.regplot(x=col,y=tar_col,data=df)
    plt.title(col)
    plt.show()

# %%
for col in cat_cols:
    sns.violinplot(y=col,x=tar_col,data=df)
    plt.title(col)
    plt.show()

# %% [markdown]
# Multivariate Analysis

# %%
full_num_cols = df.select_dtypes(include=["int64","float64"]).columns
sns.heatmap(df[full_num_cols].corr(),annot=True,fmt=".2f",cmap="Blues")

# %% [markdown]
# Preprocessing 

# %%
import sklearn
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error,r2_score
import optuna

# %%
print("scikit-learn=="+sklearn.__version__)
print("xgboost=="+xgboost.__version__)
print("optuna=="+optuna.__version__)

# %%
df.sample(1)

# %%
df = df.drop(columns=["match_id"])

# %%
num_cols = df.select_dtypes(include=["int64","float64"]).drop(columns=["final_score"]).columns

# %%
df.sample(1)

# %%
x = df.drop(columns=["final_score"])
y = df["final_score"]

# %%
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# %%
num_pipeline = Pipeline(steps=[
    ("scaling",StandardScaler())
])

# %%
cat_pipeline = Pipeline(steps=[
    ("oh_encoding",OneHotEncoder())
])

# %%
preprocessing = ColumnTransformer(transformers=[
    ("num_pipeline",num_pipeline,num_cols),
    ("cat_pipeline",cat_pipeline,cat_cols)
])

# %%
def objective(trial):
    model_name = trial.suggest_categorical("regressor", ["lr", "dt", "rf", "xgb"])

    if model_name == "lr":
        model = LinearRegression()

    elif model_name == "dt":
        model = DecisionTreeRegressor(
        max_depth=trial.suggest_int("max_depth", 3, 20),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10)
    )

    elif model_name == "rf":
        model = RandomForestRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        max_depth=trial.suggest_int("max_depth", 5, 30),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        n_jobs=-1
    )

    elif model_name == "xgb":
        model = XGBRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        gamma=trial.suggest_float("gamma", 0, 5),
        reg_alpha=trial.suggest_float("reg_alpha", 0, 5),
        reg_lambda=trial.suggest_float("reg_lambda", 0, 5),
        n_jobs=-1,
        verbosity=0
    )

    pipe = Pipeline(steps=[
        ("preprocessing",preprocessing),
        ("final_model",model)
    ])

    score = cross_val_score(pipe,x_train,y_train,cv=5,scoring="neg_mean_squared_error")
    return score.mean()

# %%
study = optuna.create_study(direction="maximize")
study.optimize(objective,n_trials=100)

# %%
params = study.best_params
print(params)

# %%
params = study.best_params
model_name = params["regressor"]

if model_name == "lr":
    final_model = LinearRegression()

elif model_name == "dt":
    final_model = DecisionTreeRegressor(
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"]
    )

elif model_name == "rf":
    final_model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        n_jobs=-1
    )

elif model_name == "xgb":
    final_model = XGBRegressor(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        gamma=params["gamma"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        n_jobs=-1,
        verbosity=0
    )


# %%
final_pipeline = Pipeline(steps=[
    ("preprocessing",preprocessing),
    ("final_model",final_model)
])

# %%
final_pipeline.fit(x_train,y_train)

# %%
y_train_pred = final_pipeline.predict(x_train)
y_test_pred = final_pipeline.predict(x_test)

# %%
train_rmse = root_mean_squared_error(y_train,y_train_pred)
print(f"Train RMSE: {train_rmse}")
test_rmse = root_mean_squared_error(y_test,y_test_pred)
print(f"Test RMSE: {test_rmse}")

# %%
train_nrmse_range = train_rmse/(y_train.max()-y_train.min())
print(f"Train NRMSE range: {train_nrmse_range}")
test_nrmse_range = test_rmse/(y_test.max()-y_test.min())
print(f"Test NRMSE range: {test_nrmse_range}")

# %%
train_nrmse_mean = train_rmse/y_train.mean()
print(f"Train NRMSE Mean: {train_nrmse_mean}")
test_nrmse_mean = test_rmse/y_test.mean()
print(f"Test NRMSE Mean: {test_nrmse_mean}")

# %%
train_nrmse_std = train_rmse/y_train.std()
print(f"Train NRMSE std: {train_nrmse_std}")
test_nrmse_std = test_rmse/y_test.std()
print(f"Test NRMSE std: {test_nrmse_std}")

# %%
train_r2 = r2_score(y_train,y_train_pred)
print(f"Train r2: {train_r2}")
test_r2 = r2_score(y_test,y_test_pred)
print(f"Test r2: {test_r2}")

# %% [markdown]
# Saving Model(using joblib) as .pkl

# %%
import joblib

# %%
print("joblib=="+joblib.__version__)

# %%
joblib.dump(final_pipeline,"final_pipeline.pkl")

# %%
import shap

# %%
print("shap=="+shap.__version__)

# %%
shap_preprocessor = final_pipeline.named_steps["preprocessing"]
shap_model = final_pipeline.named_steps["final_model"]

# %%
shap_features = []
for col in shap_preprocessor.get_feature_names_out():
    shap_features.append(col.split("__")[-1])

# %%
shap_features

# %%
x_test_t = pd.DataFrame(
    shap_preprocessor.transform(x_test),
    columns=shap_features
)

# %%
explainer = shap.TreeExplainer(shap_model)

# %%
shap_values = explainer(x_test_t)

# %%
shap_values.shape

# %%
shap.plots.beeswarm(shap_values)

# %%
shap.plots.bar(shap_values)

# %%



