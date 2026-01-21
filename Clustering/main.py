from os import wait
import pandas as pd 
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns


df = pd.read_csv('parkinsons_updrs.data', sep=',')

print(df.head(10))

cols_to_drop = ["subject#", "test_time", "motor_UPDRS", "total_UPDRS"]

X = df.drop(columns=cols_to_drop)

print(X.columns)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print(np.mean(X_scaled[:, 0]))
print(np.std(X_scaled[:, 0]))

inertia = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=2137)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(k_values, inertia, marker='o')
plt.xlabel("Liczba klasterow k")
plt.ylabel("Inertia")
plt.title("Metoda łokcia - K-Means")
plt.show()

for k in [2,3,4,5]:
    kmeans = KMeans(n_clusters=k, random_state=2137)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"k={k}, silhouette_score = {score:.3f}")


kmeans = KMeans(n_clusters=2, random_state=2137)
labels = kmeans.fit_predict(X_scaled)

df["cluster"] = labels

print(df.groupby("cluster")[["motor_UPDRS", "total_UPDRS"]].mean())

plt.figure(figsize=(6, 4))
sns.stripplot(x="cluster", y="total_UPDRS", data=df)
plt.title("total_UPDRS - rozrzut w klastrach")
plt.xlabel("Klaster")
plt.ylabel("total_UPDRS")
plt.tight_layout()
plt.show()


params = [
        (0.5, 5),
        (1.0, 5),
        (1.5, 5),
        (1.0, 10)
        ]

for eps, min_samples in params:
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if n_clusters > 1 and -1 not in labels:
        score = silhouette_score(X_scaled, labels)
        print(f"eps={eps}, min_samples={min_samples}, clusters={n_clusters}, silhouette={score:.3f}")
    else:
        print(f"eps={eps}, min_samples={min_samples}, clusters={n_clusters}, (brak silhouette)")
