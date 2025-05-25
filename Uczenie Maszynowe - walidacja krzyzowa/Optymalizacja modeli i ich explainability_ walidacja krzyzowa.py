import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import h2o
from h2o.automl import H2OAutoML
from sklearn.metrics import accuracy_score
import shap

#  Wczytanie i wstępne przygotowanie danych

df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# Tworzenie pierwszych pipeline-ów

numeric_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('onehot', OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Tworzenie pierwszych pipeline-ów c.d.

def build_pipeline(model):
    return Pipeline(steps=[("preprocessor", preprocessor),
                           ("classifier", model)])

log_reg = LogisticRegression(max_iter=1000)
log_reg_pipeline = build_pipeline(log_reg)

rf = RandomForestClassifier(random_state=42)
rf_pipeline = build_pipeline(rf)

svm = SVC()
svm_pipeline = build_pipeline(svm)

# Grid search CV dla Logistic Regression

param_grid = {
    'classifier__C': [0.1, 1.0, 10.0],
    'classifier__penalty': ['l2']
}

grid_search_log_reg = GridSearchCV(log_reg_pipeline, param_grid, cv=5)
grid_search_log_reg.fit(X_train, y_train)
lr_best = grid_search_log_reg.best_params_

# Random search CV dla lasu losowego

param_dist = {
    'classifier__n_estimators': [10, 50, 100],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]

}

random_search_rf = RandomizedSearchCV(rf_pipeline, param_distributions=param_dist, n_iter=10, random_state=42, cv=5)
random_search_rf.fit(X_train, y_train)
rf_best = random_search_rf.best_params_

#  Grid search CV dla SVM

param_grid_svm = {
    'classifier__C': [0.1, 1.0, 10.0],
    'classifier__gamma': [1, 0.1, 0.01],
    'classifier__kernel': ['rbf', 'linear']
}

grid_search_svm = GridSearchCV(svm_pipeline, param_grid_svm, cv=5)
grid_search_svm.fit(X_train, y_train)
svm_best = grid_search_svm.best_params_

# AutoML

h2o.init()

train_h2o = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
test_h2o = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))

train_h2o['target'] = train_h2o['target'].asfactor()
test_h2o['target'] = test_h2o['target'].asfactor()

# AutoML cd.

aml = H2OAutoML(max_runtime_secs=60, seed=42, nfolds=5)
aml.train(y="target", training_frame=train_h2o)

lb = aml.leaderboard

# Porównanie modeli

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import h2o
from h2o.automl import H2OAutoML
from sklearn.metrics import accuracy_score
import shap

#  Wczytanie i wstępne przygotowanie danych

df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# Tworzenie pierwszych pipeline-ów

numeric_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('onehot', OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Tworzenie pierwszych pipeline-ów c.d.

def build_pipeline(model):
    return Pipeline(steps=[("preprocessor", preprocessor),
                           ("classifier", model)])

log_reg = LogisticRegression(max_iter=1000)
log_reg_pipeline = build_pipeline(log_reg)

rf = RandomForestClassifier(random_state=42)
rf_pipeline = build_pipeline(rf)

svm = SVC()
svm_pipeline = build_pipeline(svm)

# Grid search CV dla Logistic Regression

param_grid = {
    'classifier__C': [0.1, 1.0, 10.0],
    'classifier__penalty': ['l2']
}

grid_search_log_reg = GridSearchCV(log_reg_pipeline, param_grid, cv=5)
grid_search_log_reg.fit(X_train, y_train)
lr_best = grid_search_log_reg.best_params_

# Random search CV dla lasu losowego

param_dist = {
    'classifier__n_estimators': [10, 50, 100],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]

}

random_search_rf = RandomizedSearchCV(rf_pipeline, param_distributions=param_dist, n_iter=10, random_state=42, cv=5)
random_search_rf.fit(X_train, y_train)
rf_best = random_search_rf.best_params_

#  Grid search CV dla SVM

param_grid_svm = {
    'classifier__C': [0.1, 1.0, 10.0],
    'classifier__gamma': [1, 0.1, 0.01],
    'classifier__kernel': ['rbf', 'linear']
}

grid_search_svm = GridSearchCV(svm_pipeline, param_grid_svm, cv=5)
grid_search_svm.fit(X_train, y_train)
svm_best = grid_search_svm.best_params_

# AutoML

h2o.init()

train_h2o = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
test_h2o = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))

train_h2o['target'] = train_h2o['target'].asfactor()
test_h2o['target'] = test_h2o['target'].asfactor()

# AutoML cd.

aml = H2OAutoML(max_runtime_secs=60, seed=42, nfolds=5)
aml.train(y="target", training_frame=train_h2o)

lb = aml.leaderboard

# Porównanie modeli

accuracy_scores = {
    'Logistic Regression': accuracy_score (y_test, grid_search_log_reg.best_estimator_.named_steps["classifier"].predict(X_test)),
    'Random Forest': accuracy_score (y_test, random_search_rf.best_estimator_.named_steps ["classifier"].predict(X_test)),
    'SVM': accuracy_score (y_test, grid_search_svm.best_estimator_.named_steps["classifier"].predict(X_test)),
    'H20 AutoML': accuracy_score (y_test, aml.leader.predict(test_h2o).as_data_frame()["predict"])
}

accuracy_df = pd.DataFrame(accuracy_scores.items(), columns=["Model", "Accuracy"])

