from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler as scl
from scipy.spatial import distance_matrix  # scipy.spatial.distnace_matrix
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import numpy as np
#####
X , Y = make_blobs(50 , centers=[[1,2],[3,2],[5,5],[6,4],[2,1]]) # make clusters
model = AgglomerativeClustering(
                                n_clusters = 5,          # number of clusters
                                affinity='euclidean',    # type of distance methode --> euclidean , manhatan , cosine
                                linkage='ward'         
                                # type of two clusters distance methode:
                                #^^^ average --> uses the average of the distances of each observation of the two sets.
                                #^^^ complete --> uses the maximum distances between all observations of the two sets.
                                #^^^ single --> uses the minimum of the distances between all observations of the two sets.
                                #^^^ ward --> uses Ward variance minimization algorithm
                                
)
#####
scaler = scl().fit(X)
X = scaler.transform(X)
model.fit(X)
distances = distance_matrix(X,X)
#####
X1 = hierarchy.linkage(distances,method='ward',metric='euclidean')
den = hierarchy.dendrogram(X1)
plt.show()
print(model.labels_)
