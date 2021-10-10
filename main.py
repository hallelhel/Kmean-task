from flask import Flask, render_template, request,jsonify
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np


app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/update_cluster',methods=['POST'])
def update_cluster():
  n_cluster = request.form['number']
  print(n_cluster)
  if n_cluster:
    k = n_cluster

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
    plt.figure(figsize=(10, 9))
    bottom = 0.35

    for i in range(k):
      plt.subplots_adjust(bottom)
      plt.subplot(4, 4, i + 1)
      plt.title('Number:{}'.format(reference_labels[i]), fontsize=17)
      plt.imshow(centroids[i])
      plt.show()
      plt.savefig('{}.png'.format(reference_labels[i]))


    return jsonify({'k': k})
  return jsonify({'error': 'Please enter a digit'})






if __name__ == "__main__":
  app.run(debug=True)