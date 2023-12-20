import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

from QuestionFour import data_prep
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class Module(object):
    """
    Base class defining the structure and interface for neural network modules
    with placeholders for forward and backward computations.
    """

    def __init__(self):
        self.gradInput = None  # stores gradient
        self.output = None  # stores loss

    def forward(self, *input):
        """
        Placeholder for forward pass. Defines the computation performed at every call.
        Enforces that subclasses must implement their own version of the forward method
        """
        raise NotImplementedError

    def backward(self, *input):
        """
        Placeholder for backward pass. Defines the computation performed at every call.
        Enforces that subclasses must implement their own version of the backward method
        """
        raise NotImplementedError


class Linear(Module):
    """
    The input is supposed to have two dimensions (batchSize, in_feature)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features  # dimensions
        self.out_features = out_features  # dimensions
        self.weight = math.sqrt(1. / (out_features * in_features)) * np.random.randn(out_features, in_features)
        self.bias = np.zeros(out_features)
        self.gradWeight = None
        self.gradBias = None

    def forward(self, x):  # this is our linear unit
        self.output = np.dot(x, self.weight.transpose()) + np.repeat(self.bias.reshape([1, -1]), x.shape[0], axis=0)
        return self.output

    def backward(self, x, gradOutput):
        self.gradInput = np.dot(gradOutput, self.weight)
        self.gradWeight = np.dot(gradOutput.transpose(), x)
        self.gradBias = np.sum(gradOutput, axis=0)
        return self.gradInput

    def gradientStep(self, lr):
        self.weight = self.weight - lr * self.gradWeight
        self.bias = self.bias - lr * self.gradBias


class ReLU(Module):
    """
    Implement the Rectified Linear Unit activation function for introducing non-linearity in the network.
    """

    def __init__(self, bias=True):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.output = x.clip(0)
        return self.output

    def backward(self, x, gradOutput):
        self.gradInput = (x > 0) * gradOutput
        return self.gradInput


class MLP(Module):

    def __init__(self, input_features, hidden_size_1, hidden_size_2, num_classes=52):
        super(MLP, self).__init__()
        self.fc1 = Linear(input_features, hidden_size_1)
        self.relu1 = ReLU()
        self.fc2 = Linear(hidden_size_1, hidden_size_2)
        self.relu2 = ReLU()
        self.fc3 = Linear(hidden_size_2, num_classes)

    def forward(self, x):
        x = self.fc1.forward(x)
        x = self.relu1.forward(x)
        x = self.fc2.forward(x)
        x = self.relu2.forward(x)
        x = self.fc3.forward(x)
        return x

    def backward(self, x, gradient):
        gradient = self.fc3.backward(self.relu2.output, gradient)
        gradient = self.relu2.backward(self.fc2.output, gradient)
        gradient = self.fc2.backward(self.relu1.output, gradient)
        gradient = self.relu1.backward(self.fc1.output, gradient)
        gradient = self.fc1.backward(x, gradient)
        return gradient

    def gradientStep(self, lr):
        self.fc3.gradientStep(lr)
        self.fc2.gradientStep(lr)
        self.fc1.gradientStep(lr)
        return True


class CrossEntropyCriterion(Module):
    """
    This implementation of the cross-entropy loss assumes that the data comes as a 2 dimensional array
    of size (batch_size,num_classes) and the labels as a vector of size (num_classes)
    """

    def __init__(self, num_classes=52):
        super(CrossEntropyCriterion, self).__init__()
        self.num_classes = num_classes

    def forward(self, x, labels):
        target = np.zeros([x.shape[0], self.num_classes])
        for i in range(x.shape[0]):
            target[i, labels[i]] = 1
        self.output = -np.sum(target * np.log(np.abs(x) + 1e-8))
        return self.output

    def backward(self, x, labels):
        self.gradInput = x
        for i in range(x.shape[0]):
            self.gradInput[i, labels[i]] = x[i, labels[i]] - 1
        return self.gradInput


def train_model(num_epochs, learn_rate, batch_size, model, criterion, train_data, train_labels, val_data, val_labels):
    n_train, n_val = len(train_data), len(val_data)
    train_loss = np.empty([num_epochs, int(n_train / batch_size)])
    val_loss = np.empty([num_epochs, int(n_val / batch_size)])

    for epoch in range(num_epochs):

        # Training loop
        for i in range(int(n_train / batch_size)):
            x = train_data[batch_size * i:batch_size * (i + 1)]
            y = train_labels[batch_size * i:batch_size * (i + 1)]
            y_pred = model.forward(x)
            train_loss[epoch, i] = criterion.forward(y_pred, y)
            grad0 = criterion.backward(y_pred, y)
            grad = model.backward(x, grad0)
            model.gradientStep(learn_rate)

            # Validation loop
        for j in range(int(n_val / batch_size)):
            x = val_data[batch_size * j:batch_size * (j + 1)]
            y = val_labels[batch_size * j:batch_size * (j + 1)]
            y_pred = model.forward(x)
            val_loss[epoch, j] = criterion.forward(y_pred, y)

        if (epoch + 1) % 10 == 0:
            print('Training epoch:', epoch + 1)

    # Plot output, if desired
    plt.plot(np.mean(train_loss, axis=1))
    plt.plot(np.mean(val_loss, axis=1))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train', 'val'])
    plt.show()


def evaluate_model(model, val_data, val_labels, batch_size, num_samples):
    n_val = len(val_data)
    y_pred = np.empty([int(n_val / batch_size), batch_size])  # if n_val/batch_size is not an integer it doesn't work

    for i in range(int(n_val / batch_size)):
        x = val_data[batch_size * i:batch_size * (i + 1)]
        y = val_labels[batch_size * i:batch_size * (i + 1)]
        y_pred[i, :] = np.argmax(model.forward(x), axis=1)

    rand_index = np.random.randint(len(val_data), size=num_samples)
    model_accuracy = np.mean((y_pred.flatten()[rand_index] == val_labels[rand_index]))

    return model_accuracy


def train_original_features(data, required_cols, target, LabelEncoder, num_samples, num_epochs, learn_rate,
                            batch_size):

    dataRequired = data[required_cols]
    X = dataRequired[predictors].to_numpy()
    y = LabelEncoder.fit_transform(dataRequired[target].to_numpy())
    train_data, train_labels, val_data, val_labels = data_setup(X, y)
    model = MLP(input_features=10, hidden_size_1=32, hidden_size_2=64)
    criterion = CrossEntropyCriterion()
    train_model(num_epochs, learn_rate, batch_size, model, criterion, train_data, train_labels, val_data, val_labels)
    model_accuracy = evaluate_model(model, val_data, val_labels, batch_size, num_samples)
    print('Model accuracy:', model_accuracy)


def train_pca_features(num_samples, num_epochs, learn_rate,
                            batch_size, train_data, train_labels, val_data, val_labels):
    model = MLP(input_features=3, hidden_size_1=32, hidden_size_2=64)
    criterion = CrossEntropyCriterion()
    train_model(num_epochs, learn_rate, batch_size, model, criterion, train_data, train_labels, val_data, val_labels)
    model_accuracy = evaluate_model(model, val_data, val_labels, batch_size, num_samples)
    print('Model accuracy:', model_accuracy)


def data_setup(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_data = X_train
    train_labels = y_train
    val_data = np.delete(X_val, -1, axis=0)  # This is done to solve a bug in evaluate_model that causes index overflow
    val_labels = np.delete(y_val, -1, axis=0)
    return train_data, train_labels, val_data, val_labels


if __name__ == "__main__":
    num_samples = int(1e3)
    num_epochs = 50
    learn_rate = 1e-3
    batch_size = 10
    LabelEncoder = LabelEncoder()
    data = data_prep()
    required_cols = ['duration', "danceability", 'energy',
                     'loudness', 'speechiness', 'acousticness',
                     'instrumentalness', 'liveness', 'valence', 'tempo', 'track_genre']

    target = 'track_genre'
    predictors = ['duration', "danceability", 'energy',
                  'loudness', 'speechiness', 'acousticness',
                  'instrumentalness', 'liveness', 'valence', 'tempo']
    train_original_features(data, required_cols, target, LabelEncoder, num_samples, num_epochs, learn_rate, batch_size)
    PCA = pd.read_csv("pca_features.csv")
    X = PCA.values
    y = LabelEncoder.fit_transform(data["track_genre"].to_numpy())
    train_data, train_labels, val_data, val_labels = data_setup(X, y)
    train_pca_features(num_samples, num_epochs, learn_rate, batch_size, train_data, train_labels, val_data, val_labels)
