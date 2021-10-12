from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
from skimage import color
from skimage import io
# from google.colab import files
import math


def retrieve_info(cluster_labels, y_test):
    # Initializing
    reference_labels = {}
    # For loop to run through each label of cluster label
    for i in range(len(np.unique(kmeans.labels_))):
        index = np.where(cluster_labels == i, 1, 0)
        num = np.bincount(y_test[index == 1]).argmax()
        reference_labels[i] = num
    return reference_labels


(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalization of ‘x_train’
x_test_as_mat = x_test
x_test = x_test.astype('float32')
x_test = x_test / 255.0
# Reshaping of ‘x_train’
x_test = x_test.reshape(10000, 28 * 28)
# Training the model
k = 4
kmeans = KMeans(n_clusters=k)
kmeans.fit(x_test)
reference_labels = retrieve_info(kmeans.labels_, y_test)
# ‘number_labels’ is a list which denotes the number displayed in image
# number_labels = np.random.rand(len(kmeans.labels_))
# for i in range(len(kmeans.labels_)):
#     number_labels[i] = reference_labels[kmeans.labels_[i]]

# Cluster centroids is stored in ‘centroids’
centroids = kmeans.cluster_centers_

centroids = centroids.reshape(k, 28, 28)
centroids = centroids * 255
fig = plt.figure(figsize=(10, 9))

# bottom = 0.35
# row = math.floor(math.sqrt(k))
# col = math.ceil(math.sqrt(k))
for i in range(k):
    # plt.subplots_adjust(bottom)
    sub = fig.add_subplot(k,4,i+1)
    # plt.subplot(4, 4, i + 1)
    sub.set_title('Number:{}'.format(reference_labels[i]), fontsize=12)
    sub.imshow(centroids[i])
    plt.axis('off')
    # plt.savefig('{}.png'.format(reference_labels[i]))
    # plt.show()
plt.savefig('cluster.png')
plt.show()


# def five_more_img():
indexes = {}
for i in range(k):
    array_of_index = np.where(kmeans.labels_ == i)
    indexes[i] = array_of_index[0][:5]
    # return indexes

indexes_of_img_to_show = indexes

fig_5 = plt.figure(figsize=(10, 9))

loc = 0
for i in range(k):
    for j in indexes_of_img_to_show[i]:
        sub_5 = fig_5.add_subplot(k, 5, loc + 1)
        sub_5.set_title('Number:{}'.format(reference_labels[i]), fontsize=12)
        sub_5.imshow(x_test_as_mat[j])
        plt.axis('off')
        loc=loc+1
plt.savefig('5_img.png')
plt.show()

