from markupsafe import escape_silent
from matplotlib.typing import LineStyleType
import numpy as np 
import matplotlib.pyplot as plt 
from pandas.core.common import random_state
from sklearn.datasets import load_breast_cancer, load_iris, load_wine 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Breast Cancer
#X, y = load_breast_cancer(return_X_y=True)

# Iris
#X, y = load_iris(return_X_y=True)

# Wine:
X, y = load_wine(return_X_y=True)


X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2137
        )

#rf 

rf = RandomForestClassifier(n_estimators=100, random_state=2137)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f'rf acc {rf_acc}')


#xgboost

xgb = XGBClassifier(
        eval_metric='logloss',
        random_state=2137
        )

xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)

print(f'xgb acc {xgb_acc}')

param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.5, 0.7, 1.0]
        }

grid = GridSearchCV(
        XGBClassifier(eval_metric='logloss'),
        param_grid,
        scoring='accuracy',
        cv=3,
        n_jobs=1
        )

grid.fit(X_train, y_train)

print(f'Best params: {grid.best_params_}')
best_xgb = grid.best_estimator_


tuned_pred = best_xgb.predict(X_test)
tuned_xgb_acc = accuracy_score(y_test, tuned_pred)
print(f'Tuned xgb acc {tuned_xgb_acc}')

estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svc', SVC(probability=True)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('tree', DecisionTreeClassifier(max_depth=5))
]

stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)

stack.fit(X_train, y_train)
stack_pred = stack.predict(X_test)
stack_acc = accuracy_score(y_test, stack_pred)

print("Stacking Accuracy:", stack_acc)

models = ['Random Forest', 'XGBoost', 'XGBoost tuned', 'Stacking']
accuracies = [rf_acc, xgb_acc, tuned_xgb_acc, stack_acc]

plt.figure(figsize=(8,6))
plt.bar(models, accuracies)
plt.ylabel('Accuracy')
plt.title('Porównanie dokładności modeli')
plt.ylim(0.8, 1.0)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

