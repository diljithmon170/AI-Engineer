from sklearn.cluster import KMeans
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
# print(df.head())

df = df.drop(['target', 'sepal width (cm)', 'sepal length (cm)'], axis='columns')
# print(df.head())

# plotting the data
plt.scatter(df['petal length (cm)'], df['petal width (cm)'])
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()

# # preprocessing minmax scaler
# scaler = MinMaxScaler()

# scaler.fit(df[['petal length (cm)']])
# df['petal length (cm)'] = scaler.transform(df[['petal length (cm)']])
# scaler.fit(df[['petal width (cm)']])
# df['petal width (cm)'] = scaler.transform(df[['petal width (cm)']])
# print(df.head())

# plt.scatter(df['petal length (cm)'], df['petal width (cm)'])
# plt.show()

# kmeans clustering
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['petal length (cm)', 'petal width (cm)']])
# print(y_predicted)
df['cluster'] = y_predicted
# print(df.head())

# km centers
print(km.cluster_centers_)

# plotting the clusters
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color='green')
plt.scatter(df2['petal length (cm)'], df2['petal width (cm)'], color='red')
plt.scatter(df3['petal length (cm)'], df3['petal width (cm)'], color='black')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='purple', marker='*', label='centroid')
plt.legend()
plt.show()


# elbow point finding
sse = []
k_rng = range(1, 10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['petal length (cm)', 'petal width (cm)']])
    sse.append(km.inertia_)
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng, sse)
plt.show()