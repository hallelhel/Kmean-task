from flask import Flask, render_template, request,jsonify
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import io
from base64 import encodebytes
from PIL import Image



app = Flask(__name__)


@app.route('/')
def index():
  return render_template('index.html')



@app.route('/update_cluster',methods=['POST'])
def update_cluster():
  n_cluster = request.form['number']
  print(n_cluster)
  if n_cluster:
    k = int(n_cluster)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test_as_mat = x_test

    # Normalization of ‘x_test’
    x_test = x_test.astype('float32')
    x_test = x_test / 255.0
    # Reshaping of ‘x_test’
    x_test = x_test.reshape(10000, 28 * 28)
    # Training the model
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(x_test)
    reference_labels = retrieve_info(kmeans, kmeans.labels_, y_test)

    # Cluster centroids is stored in ‘centroids’
    centroids = kmeans.cluster_centers_
    centroids = centroids.reshape(k, 28, 28) # to see centroid as image
    centroids = centroids * 255 # nullify the normalization effect

    # for 5 other images
    indexes_of_img_to_show = {}
    for i in range(k):
      array_of_index = np.where(kmeans.labels_ == i)
      indexes_of_img_to_show[i] = array_of_index[0][:5]

    loc = 0
    fig = plt.figure(figsize=(10, 9))

    for i in range(k):
      sub = fig.add_subplot(k, 6, loc+1)
      sub.set_title('label:{} center image'.format(reference_labels[i]), fontsize=10,loc='left',pad=-0.1)
      sub.imshow(centroids[i])
      plt.axis('off')
      loc = loc + 1

      for j in indexes_of_img_to_show[i]:
        sub = fig.add_subplot(k, 6, loc + 1)
        sub.set_title('label:{}'.format(reference_labels[i]), fontsize=8)
        sub.imshow(x_test_as_mat[j])
        plt.axis('off')
        loc = loc + 1
    fig.tight_layout()
    plt.savefig('cluster.png')
    image_path = 'cluster.png'
    encoded_img = get_response_image(image_path)
    return jsonify({'image': encoded_img})

  return jsonify({'error': 'Please enter a digit'})


'''
    Associates most probable label with each cluster in KMeans model
    returns: dictionary of clusters assigned to each label
    '''


def retrieve_info(kmeans, cluster_labels, y_test):
  # Initializing
  reference_labels = {}
  # For loop to run through each label of cluster label
  for i in range(len(np.unique(kmeans.labels_))):
    index = np.where(cluster_labels == i, 1, 0)
    num = np.bincount(y_test[index == 1]).argmax()
    reference_labels[i] = num
  return reference_labels

def get_response_image(image_path):
  pil_img = Image.open(image_path, mode='r')  # reads the PIL image
  byte_arr = io.BytesIO()
  pil_img.save(byte_arr, format='PNG')  # convert the PIL image to byte array
  encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii')  # encode as base64
  return encoded_img



if __name__ == "__main__":
  app.run(debug=True)