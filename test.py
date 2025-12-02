from sklearn.cluster import SpectralCoclustering
import numpy as np

file = 'facebookG.txt'
X = np.array(np.loadtxt(file))


clustering = SpectralCoclustering(n_clusters=10, random_state=0).fit(X)
print(clustering)