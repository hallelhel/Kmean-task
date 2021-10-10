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


def retrieve_info(cluster_labels, y_train):
    # Initializing
    reference_labels = {}
    # For loop to run through each label of cluster label
    for i in range(len(np.unique(kmeans.labels_))):
        index = np.where(cluster_labels == i, 1, 0)
        num = np.bincount(y_train[index == 1]).argmax()
        reference_labels[i] = num
    return reference_labels


(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalization of ‘x_train’
x_train = x_train.astype('float32')
x_train = x_train / 255.0
# Reshaping of ‘x_train’
x_train = x_train.reshape(60000, 28 * 28)
# Training the model
k = 10
kmeans = KMeans(n_clusters=k)
kmeans.fit(x_train)
reference_labels = retrieve_info(kmeans.labels_, y_train)
number_labels = np.random.rand(len(kmeans.labels_))
for i in range(len(kmeans.labels_)):
    number_labels[i] = reference_labels[kmeans.labels_[i]]

# Cluster centroids is stored in ‘centroids’
centroids = kmeans.cluster_centers_

centroids = centroids.reshape(k, 28, 28)
centroids = centroids * k
fig = plt.figure(figsize=(10, 9))
bottom = 0.35
row = math.floor(math.sqrt(k))
col = math.ceil(math.sqrt(k))
for i in range(k):
    # plt.subplots_adjust(bottom)
    sub = fig.add_subplot(k,4,i+1)
    # plt.subplot(4, 4, i + 1)
    sub.set_title('Number:{}'.format(reference_labels[i]), fontsize=12)
    sub.imshow(centroids[i])
    # plt.savefig('{}.png'.format(reference_labels[i]))
    # plt.show()
plt.show()







#
#
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# #
# # print(type(x_train))
# # print(type(x_test))
# # print(type(y_train))
# # print(type(y_test))
# #
# # print(x_train.shape)
# # print(x_test.shape)
# # print(y_train.shape)
# # print(y_test.shape)
# #
# plt.gray() # B/W Images
# plt.figure(figsize = (10,9)) # Adjusting figure size
# # Displaying a grid of 3x3 images
# for i in range(9):
#  plt.subplot(3,3,i+1)
#  plt.imshow(x_train[i])
#  plt.show()
#
# for i in range(5):
#   print(y_train[i])
#
#
# # Data Normalization
# # Conversion to float
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# # Normalization
# x_train = (x_train)/255.0
# x_test = (x_test)/255.0
#
# X_train = x_train.reshape(len(x_train),-1)
# X_test = x_test.reshape(len(x_test),-1)
#
# total_clusters = len(np.unique(y_test))
# # total_clusters = 9
# # Initialize the K-Means model
# kmeans = KMeans(n_clusters = total_clusters)
# # Fitting the model to training set
# kmeans.fit(X_train)
#
# # print(kmeans.labels_)
#
# '''
#  Associates most probable label with each cluster in KMeans model
#  returns: dictionary of clusters assigned to each label
# '''
# def retrieve_info(cluster_labels,y_train):
#     # Initializing
#     reference_labels = {}
#     # For loop to run through each label of cluster label
#     for i in range(len(np.unique(kmeans.labels_))):
#         index = np.where(cluster_labels == i,1,0)
#         num = np.bincount(y_train[index==1]).argmax()
#         reference_labels[i] = num
#     return reference_labels
#
# reference_labels = retrieve_info(kmeans.labels_,y_train)
# number_labels = np.random.rand(len(kmeans.labels_))
# for i in range(len(kmeans.labels_)):
#   number_labels[i] = reference_labels[kmeans.labels_[i]]
# print(number_labels[:20].astype('int'))
# print(y_train[:20])
#
# # Testing model on Testing set
# # Initialize the K-Means model
# kmeans = KMeans(n_clusters = 256)
# # Fitting the model to testing set
# kmeans.fit(X_test)
# reference_labels = retrieve_info(kmeans.labels_,y_test)
# number_labels = np.random.rand(len(kmeans.labels_))
# for i in range(len(kmeans.labels_)):
#     number_labels[i] = reference_labels[kmeans.labels_[i]]
#
# # Cluster centroids is stored in ‘centroids’
# centroids = kmeans.cluster_centers_
# centroids = centroids.reshape(256,28,28)
# centroids = centroids * 255
# plt.figure(figsize = (10,9))
# bottom = 0.35
# for i in range(16):
#  plt.subplots_adjust(bottom)
#  plt.subplot(4,4,i+1)
#  plt.title('Number:{}'.format(reference_labels[i]),fontsize = 17)
#  plt.imshow(centroids[i])
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# # Normalization of ‘x_train’
# x_train = x_train.astype('float32')
# x_train = x_train/255.0
# # Reshaping of ‘x_train’
# x_train = x_train.reshape(60000,28*28)
# # Training the model
# n_clusters = 9
# kmeans = KMeans(n_clusters= n_clusters)
# kmeans.fit(x_train)
# reference_labels = retrieve_info(kmeans.labels_,y_train)
# number_labels = np.random.rand(len(kmeans.labels_))
# for i in range(len(kmeans.labels_)):
#   number_labels[i] = reference_labels[kmeans.labels_[i]]
# # predicted_cluster = kmeans.predict(image)
# # number_labels[[predicted_cluster]]
#
# # Cluster centroids is stored in ‘centroids’
# centroids = kmeans.cluster_centers_
#
# centroids = centroids.reshape(n_clusters,28,28)
# centroids = centroids * n_clusters
# plt.figure(figsize = (10,9))
# bottom = 0.35
# for i in range(n_clusters):
#  plt.subplots_adjust(bottom)
#  plt.subplot(4,4,i+1)
#  plt.title('Number:{}'.format(reference_labels[i]),fontsize = 17)
#  plt.imshow(centroids[i])
#  plt.show()