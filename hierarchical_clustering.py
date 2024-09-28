# If you need:
# !pip install scikit-learn==0.23.1

# Unsupervised hierarchical clustering
#%matplotlib inline
import scipy
import pylab
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from scipy import ndimage
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import euclidean_distances
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import jaccard_score, f1_score
from sklearn.metrics import classification_report


# Input as a csv file (all the classified data to train your model):
input = 'morphology.csv'
# The names of your columns which will be used to classify tha data
columns_x = ['log(W1)', 'log(NUV)', 'n'] # 'q', 'SPIRE_250', 'NUV/3.4', 'n', '3.4/250'
# The name of your column with the classification (the categories to me numerical integers: e.g. 0 and 1)



# Main

pdf = pd.read_csv(input)
pdf = pdf.dropna(axis=0)
pdf['log(W1)'] = np.log(pdf['WISE_3.4'].values)
pdf['log(NUV)'] = np.log(pdf['GALEX_NUV'].values)
#pdf['NUV/3.4'] = pdf.GALEX_NUV - pdf['WISE_3.4']
#pdf['3.4/250'] = pdf['WISE_3.4'] - pdf.SPIRE_250
X = pdf[columns_x].values
X = np.nan_to_num(X)

# Normalize the data:
min_max_scaler = MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(X)

# Calculate the distance matrix
dist_matrix = euclidean_distances(feature_mtx,feature_mtx)

# Or
"""
leng = feature_mtx.shape[0]
dist_matrix = scipy.zeros([leng,leng])
for i in range(leng):
    for j in range(leng):
        dist_matrix[i,j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j])
"""

# Hierarchical clustering
Z = hierarchy.linkage(dist_matrix, 'complete') # other options: 'single', 'average', 'weighted'

agglom = AgglomerativeClustering(n_clusters = 2, linkage = 'complete')
agglom.fit(dist_matrix)
pdf['cluster_'] = agglom.labels_
pdf.to_csv('clustering.csv')

print("f1-score:")
print(f1_score(pdf.type, pdf.cluster_, average='weighted'))

print("Jaccard score:")
print(jaccard_score(pdf.type, pdf.cluster_, pos_label=0))

print("Classification report:")
print(classification_report(pdf.type, pdf.cluster_))

# Or
"""
# In some applications we want a partition of disjoint clusters just as in flat clustering. So you can use a cutting line:
max_d = 3 # maximum distance
clusters = fcluster(Z, max_d, criterion='distance')

#Or
k = 5 # number of clusters
clusters = fcluster(Z, k, criterion='maxclust')
"""
"""
# Plot the dendrogram
fig = pylab.figure(figsize=(18,50))
def llf(id):
    return '[%s %s %s]' % (pdf['name'][id], pdf['n'][id], int(float(pdf['q'][id])) )

dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'right')
plt.savefig("dendrogram.png")
"""

# Plot the classified data:
n_clusters = max(agglom.labels_)+1
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))

# Create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(16,14))

for color, label in zip(colors, cluster_labels):
    subset = pdf[pdf.cluster_ == label]
    #for i in subset.index:
    #        plt.text(subset.n[i], subset.SPIRE_250[i],str(subset['name'][i]), rotation=25)
    plt.scatter(subset.n, subset.SPIRE_250, s= subset.GALEX_NUV*10, c=color, label='cluster'+str(label),alpha=0.5)
#    plt.scatter(subset.horsepow, subset.mpg)
plt.legend()
#plt.ylim(-0.1,20)
plt.xscale("log")
plt.title('Title')
plt.xlabel('Label_x')
plt.ylabel('Label_y')
plt.savefig('data_clustering.png')
