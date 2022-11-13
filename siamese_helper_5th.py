import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from uji import UJI


def make_pairs(x, y):
    """Creates a tuple containing image pairs with corresponding label.

    Arguments:
        x: List containing images, each index in this list corresponds to one image.
        y: List containing labels, each label with datatype of `int`.

    Returns:
        Tuple containing two numpy arrays as (pairs_of_samples, labels),
        where pairs_of_samples' shape is (2len(x), 2,n_features_dims) and
        labels are a binary array of shape (2len(x)).
    """

    num_classes = max(y) + 1
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

    pairs = []
    labels = []

    for idx1 in range(len(x)):
        # add a matching example
        x1 = x[idx1]
        label1 = y[idx1]
        idx2 = random.choice(digit_indices[label1])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [0]

        # add a non-matching example
        label2 = random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = random.randint(0, num_classes - 1)

        idx2 = random.choice(digit_indices[label2])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [1]

    return np.array(pairs), np.array(labels).astype("float32")


"""
Split the test pairs
"""

"""
## Visualize pairs and their labels
"""

"""
## Define the model

There are two input layers, each leading to its own network, which
produces embeddings. A `Lambda` layer then merges them using an
[Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) and the
merged output is fed to the final network.
"""


# Provided two tensors t1 and t2
# Euclidean distance = sqrt(sum(square(t1-t2)))
def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


"""
## Define the constrastive Loss
"""


def loss(margin=1):
    """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.

    Arguments:
        margin: Integer, defines the baseline for distance for which pairs
                should be classified as dissimilar. - (default is 1).

    Returns:
        'constrastive_loss' function with data ('margin') attached.
    """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the constrastive loss.

        Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.

        Returns:
            A tensor containing constrastive loss as floating point value.
        """

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    return contrastive_loss


"""
## Compile the model with the contrastive loss
"""


def SNN_KNN():
    epochs = 10
    batch_size = 2
    margin = 1  # Margin for constrastive loss.

    train_X = (np.load('Data\\Saideep\\5th\\train_X_5th.npy', allow_pickle=True).astype('float16') + 100.0) / 100.0
    train_Y = np.load('Data\\Saideep\\5th\\train_Y_5th.npy', allow_pickle=True).astype('int')
    test_X = (np.load('Data\\Saideep\\5th\\test_X_5th.npy', allow_pickle=True).astype('float16') + 100.0) / 100.0
    test_Y = np.load('Data\\Saideep\\5th\\test_Y_5th.npy', allow_pickle=True).astype('int')

    train_X = np.reshape(train_X[:, :120], (train_X.shape[0], 12, 10, 1))
    test_X = np.reshape(test_X[:, :, :120], (test_X.shape[0], test_X.shape[1], 12, 10, 1))

    # make train pairs
    pairs_train, labels_train = make_pairs(train_X, train_Y)

    x_train_1 = pairs_train[:, 0]  # x_train_1.shape is (60000, 28, 28)
    x_train_2 = pairs_train[:, 1]

    input = layers.Input((12, 10, 1))
    x = tf.keras.layers.BatchNormalization()(input)
    x = layers.Conv2D(4, (3, 3), activation="relu")(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(16, (3, 2), activation="relu")(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = layers.Dense(32, activation="relu", name="encodings")(x)
    embedding_network = keras.Model(input, x)

    input_1 = layers.Input((12, 10, 1))
    input_2 = layers.Input((12, 10, 1))

    # As mentioned above, Siamese Network share weights between
    # tower networks (sister networks). To allow this, we will use
    # same embedding network for both tower networks.
    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)

    merge_layer = layers.Lambda(euclidean_distance)([tower_1, tower_2])
    normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
    output_layer = layers.Dense(1, activation="relu")(normal_layer)
    siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)

    siamese.compile(loss=loss(margin=margin), optimizer="Adam", metrics=["accuracy"])
    siamese.summary()

    """
    ## Train the model
    """

    history = siamese.fit(
        [x_train_1, x_train_2],
        labels_train,
        batch_size=batch_size,
        epochs=epochs,
    )

    """
    ## Visualize results
    """

    """
    ## Evaluate the model
    """

    intermediate_layer_model = keras.Model(inputs=embedding_network.input,
                                           outputs=embedding_network.get_layer("encodings").output)

    snn_encodings_train = intermediate_layer_model.predict(train_X)

    neigh = KNeighborsClassifier(n_neighbors=4)
    neigh.fit(snn_encodings_train, train_Y)

    uji = UJI.from_cache("test", 1, cache_dir="uji/db_cache")
    accuracies = {}
    acc_tmp = np.zeros(15)
    # iterate over each test month
    # go over months beyond the first provided month

    for month in range(15):
        snn_encodings_tmp = intermediate_layer_model.predict(test_X[month])
        # predict test using "train_waps"
        pred_y = neigh.predict(snn_encodings_tmp)
        pred_y_coord = uji.labels_to_coords(pred_y)
        test_y_coord = uji.labels_to_coords(test_Y[month])
        # compute the distances
        dists = uji.compute_distances(pred_y_coord, test_y_coord).flatten()
        acc_tmp[month] = np.mean(dists)
        # store the distances as a 2D array
        accuracies[month + 1] = dists
        print(np.mean(dists))
    avg_err = np.sum(acc_tmp) / 15
    print('avg:')
    print(avg_err)
    np.save('results\\5th_floor\\snn_knn_classification_5th.npy', acc_tmp)

    return accuracies


def SNN_KNN_aug():
    epochs = 30
    batch_size = 2
    margin = 1  # Margin for constrastive loss.

    train_X = (np.load('Data\\Saideep\\5th\\train_X_5th.npy', allow_pickle=True).astype('float16') + 100.0) / 100.0
    train_Y = np.load('Data\\Saideep\\5th\\train_Y_5th.npy', allow_pickle=True).astype('int')
    test_X = (np.load('Data\\Saideep\\5th\\test_aug_KNN_X_5th.npy', allow_pickle=True).astype('float16') + 100.0) / 100.0
    test_Y = np.load('Data\\Saideep\\5th\\test_Y_5th.npy', allow_pickle=True).astype('int')

    train_X = np.reshape(train_X[:, :120], (train_X.shape[0], 12, 10, 1))
    test_X = np.reshape(test_X[:, :, :120], (test_X.shape[0], test_X.shape[1], 12, 10, 1))

    # make train pairs
    pairs_train, labels_train = make_pairs(train_X, train_Y)

    x_train_1 = pairs_train[:, 0]  # x_train_1.shape is (60000, 28, 28)
    x_train_2 = pairs_train[:, 1]

    input = layers.Input((12, 10, 1))
    x = tf.keras.layers.BatchNormalization()(input)
    x = layers.Conv2D(16, (3, 3), activation="relu")(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(32, (3, 2), activation="relu")(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = layers.Dense(64, activation="relu", name="encodings")(x)
    embedding_network = keras.Model(input, x)

    input_1 = layers.Input((12, 10, 1))
    input_2 = layers.Input((12, 10, 1))

    # As mentioned above, Siamese Network share weights between
    # tower networks (sister networks). To allow this, we will use
    # same embedding network for both tower networks.
    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)

    merge_layer = layers.Lambda(euclidean_distance)([tower_1, tower_2])
    normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
    output_layer = layers.Dense(1, activation="relu")(normal_layer)
    siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)

    siamese.compile(loss=loss(margin=margin), optimizer="Adam", metrics=["accuracy"])
    siamese.summary()

    """
    ## Train the model
    """

    history = siamese.fit(
        [x_train_1, x_train_2],
        labels_train,
        batch_size=batch_size,
        epochs=epochs,
    )

    """
    ## Visualize results
    """

    """
    ## Evaluate the model
    """

    intermediate_layer_model = keras.Model(inputs=embedding_network.input,
                                           outputs=embedding_network.get_layer("encodings").output)

    snn_encodings_train = intermediate_layer_model.predict(train_X)

    neigh = KNeighborsClassifier(n_neighbors=4)
    neigh.fit(snn_encodings_train, train_Y)

    uji = UJI.from_cache("test", 1, cache_dir="uji/db_cache")
    accuracies = {}
    acc_tmp = np.zeros(15)
    # iterate over each test month
    # go over months beyond the first provided month

    for month in range(15):
        snn_encodings_tmp = intermediate_layer_model.predict(test_X[month])
        # predict test using "train_waps"
        pred_y = neigh.predict(snn_encodings_tmp)
        pred_y_coord = uji.labels_to_coords(pred_y)
        test_y_coord = uji.labels_to_coords(test_Y[month])
        # compute the distances
        dists = uji.compute_distances(pred_y_coord, test_y_coord).flatten()
        acc_tmp[month] = np.mean(dists)
        # store the distances as a 2D array
        accuracies[month + 1] = dists
        print(np.mean(dists))
    avg_err = np.sum(acc_tmp) / 15
    print('avg:')
    print(avg_err)
    np.save('results\\5th_floor\\snn_knn_classification_5th.npy', acc_tmp)

    return accuracies