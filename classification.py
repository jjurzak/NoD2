from enum import unique
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score





np.random.seed(2137)


X = np.random.randn(500, 2) * [3, 1]
y = (X[:, 0] > X[:, 1]).astype(int)

mask = (y == 1)

X_imbalanced = np.vstack([X[~mask], X[mask][:30]])
y_imbalanced = np.hstack([y[~mask], y[mask][:30]])

plt.scatter(X_imbalanced[:, 0], X_imbalanced[:, 1], c=y_imbalanced, cmap='coolwarm', alpha=0.6)
plt.title("Nieregularny rozklad klas")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()


unique, counts = np.unique(y_imbalanced, return_counts=True)

print(dict(zip(unique, counts)))

X_train, X_test, y_train, y_test = train_test_split(
        X_imbalanced, y_imbalanced, test_size=0.3, random_state=2137
        ) 

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

model_balanced = LogisticRegression(class_weight='balanced')
model_balanced.fit(X_train,y_train)
y_pred_bal = model_balanced.predict(X_test)



print("Model z class_weight='normal")
print("Acc", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


print("Model z class_weight='balanced ")
print("Acc", accuracy_score(y_test, y_pred_bal))
print(classification_report(y_test, y_pred_bal))


