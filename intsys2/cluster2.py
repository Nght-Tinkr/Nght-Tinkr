
from sklearn.cluster import AffinityPropagation
from sklearn import datasets
import matplotlib.pyplot as plt
from itertools import cycle


iris = datasets.load_iris()

X = iris.data[:, :2]


af = AffinityPropagation(preference=-17.5).fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

plt.close('all')
plt.figure(1)
plt.clf()

colors = cycle('rgmbcykrgmbcykrgmbcykrgmbcyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = X[cluster_centers_indices[k]]
    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markersize=14)
    for x in X[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()