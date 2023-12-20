# Our random seed
seed = 13427256

import math
import joblib
import seaborn as sns
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import statsmodels.api as sm
import scipy.stats as stats
# import dataframe_image as dfi

from sklearn.linear_model import LinearRegression
# from tqdm import tqdm
from typing import Callable
from scipy.stats import yeojohnson
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import RidgeCV


# HELPER FUNCTIONS

def data_prep():
    # colList = addStrings(columnNameFile)
    data = pd.read_csv("spotify52kData.csv")
    datagpd = (data.groupby(["track_name", "artists"]).max().reset_index().sort_values(by="songNumber")
               .reset_index(drop=True))  # Groups identical songs on different albums by the same artist into 1
    # group
    datagpd["duration"] = datagpd["duration"] / (1000 * 60)  # convert duration to minutes from milliseconds
    return datagpd

def rms_and_r2(y_pred, y):
    """
    Computes root mean squared error and R2 score for our univariate regression model
    """
    return np.sqrt(np.mean((y - y_pred) ** 2)), r2_score(y, y_pred)


def Linear_regression(X, y, transform=True, desc=""):
    y = y.to_numpy().reshape(-1, 1)
    X = X.to_numpy().reshape(-1, 1)
    if transform:
        X = yeojohnson(X)[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmseErr, r2 = rms_and_r2(y_pred, y_test)
    print(f"RMSE error {desc} is {rmseErr}, R2: {r2}\n")
    return y_pred, y_test


def multi_regression(X, y, transform=True, desc=""):
    y = y.to_numpy().reshape(-1, 1)
    X = X.to_numpy()
    if transform:
        for col in range(X.shape[1]):
            X[:, col] = yeojohnson(X[:, col])[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RidgeCV(alphas=np.logspace(-6, 6, 13))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmseErr, r2 = rms_and_r2(y_pred, y_test)
    print(f"RMSE error {desc} is {rmseErr}, R2: {r2}\n")
    return y_pred, y_test

def plotResiduals(yPred, ytest, ax, desc=""):
    residuals = ytest - yPred
    ax.scatter(yPred, residuals)
    ax.set_title(f"Scatter plot of residuals {desc}")
    # ax.set(xlabel='predicted values', ylabel='residuals')

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
    plt.figure()
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


def train_original_features(data, predictors, target, LabelEncoder, num_samples, num_epochs, learn_rate,
                            batch_size):
    X = data[predictors].to_numpy()
    y = LabelEncoder.fit_transform(data[target].to_numpy())
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



question_list = []
[q() for q in question_list]


def question(q: Callable):
    """Wrapper function for questions in order to initialize them.

    Args:
        q (function): Function that answers a question in the Capstone Project.
    """

    def wrapped_question():
        print(f"##### Starting {q.__name__.upper()} #####")
        np.random.seed(seed)
        q()

    question_list.append(wrapped_question)

    return wrapped_question


@question
def q1():
    pass


@question
def q2():
    pass


@question
def q3():
    spotify52k = data_prep()
    q3data = spotify52k[["mode", "popularity"]]

    MajorPop = pd.value_counts(q3data[q3data["mode"] == 1].popularity.values, normalize=True)
    MinorPop = pd.value_counts(q3data[q3data["mode"] == 0].popularity.values, normalize=True)
    fig = plt.figure()
    ax = plt.subplot()
    fig.suptitle("Popularity barplot for Major & Minor key songs")
    ax.bar(x=MajorPop.index, height=MajorPop, color="blue", label="Major")

    fig.supylabel("Probability Density")
    fig.supxlabel("Popularity Measure (Higher = More popular)")

    ax.bar(x=MinorPop.index, height=MinorPop, color="orange", label="Minor", alpha=0.8)
    fig.legend(frameon=False)

    testStat, pvalue = mannwhitneyu(MinorPop, MajorPop)
    print(f"pvalue ({pvalue}) is less than our significance level 0.05, hence the null hypothesis (songs in either key"
          f"are equally is rejected.")
    plt.savefig("output/q3ProbDensity.png")


@question
def q4():
    RMSE = dict()
    R2 = dict()
    data = data_prep()
    predictors = ['duration', "danceability", 'energy',
                  'loudness', 'speechiness', 'acousticness',
                  'instrumentalness', 'liveness', 'valence', 'tempo']
    # predictors_to_not_split = ["danceability", 'energy', 'acousticness', 'tempo', 'valence']
    # transform_split_residual_comparison(data, predictors, predictors_to_not_split)
    for predictor in predictors:
        yPred, yTest = Linear_regression(data[predictor], data["popularity"], transform=True, desc="(Transformed)")
        RMSE[predictor], R2[predictor] = rms_and_r2(yPred, yTest)

    resultsTable = (pd.DataFrame(RMSE.items(),
                                 columns=["Feature", "RMSE"])
                    .sort_values(by="RMSE", ascending=False).reset_index(drop=True))
    resultsTable["R2 Score"] = [r2 for r2 in R2.values()]
    resultsTable.to_csv("q4_results.csv", index=False)


@question
def q5():
    fig = plt.figure()
    fig.supxlabel("predicted values")
    fig.supylabel("Residuals")
    ax = plt.subplot()
    data = data_prep()
    predictors = ['duration', "danceability", 'energy',
                  'loudness', 'speechiness', 'acousticness',
                  'instrumentalness', 'liveness', 'valence', 'tempo']

    yPred, yTest = multi_regression(data[predictors], data["popularity"], transform=True)
    plotResiduals(yPred, yTest, ax, desc="(for all 10 features)")
    plt.savefig("output/residualsQuestion5.png")
    print(-1)


@question
def q6():
    pass


@question
def q7():
    pass


@question
def q8():
    from sklearn.preprocessing import LabelEncoder
    LabelEncoder = LabelEncoder()
    num_samples = int(1e3)
    num_epochs = 50
    learn_rate = 1e-3
    batch_size = 10
    data = data_prep()

    target = 'track_genre'
    predictors = ['duration', "danceability", 'energy',
                  'loudness', 'speechiness', 'acousticness',
                  'instrumentalness', 'liveness', 'valence', 'tempo']
    train_original_features(data, predictors, target, LabelEncoder, num_samples, num_epochs, learn_rate, batch_size)
    PCA = pd.read_csv("pca_features.csv")
    X = PCA.values
    y = LabelEncoder.fit_transform(data["track_genre"].to_numpy())
    train_data, train_labels, val_data, val_labels = data_setup(X, y)
    train_pca_features(num_samples, num_epochs, learn_rate, batch_size, train_data, train_labels, val_data, val_labels)


@question
def q9():
    df_songs = (
        pd.read_csv("spotify52kData.csv")
        .iloc[:5000]
        .sort_values(by="popularity", ascending=False)
    )
    indices_of_duplicates = df_songs[
        df_songs.duplicated(
            keep="first", subset=["artists", "album_name", "track_name", "popularity"]
        )
    ].index.tolist()
    df_songs = df_songs.drop(indices_of_duplicates).sort_values(by="songNumber")
    df_stars = pd.read_parquet("starRatings.parquet").drop(
        columns=indices_of_duplicates
    )
    avg_star_rating = df_stars.mean(axis=0)
    num_ratings = df_stars.count(axis=0)
    df_songs["avg_star_rating"] = avg_star_rating
    df_songs["num_ratings"] = num_ratings
    df_songs["num_ratings_category"] = pd.qcut(
        df_songs["num_ratings"], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
    )

    #########
    # Is there a relationship between popularity and average star rating for the 5k songs we have explicit feedback for?
    #########

    ###
    # Using correlation to answer the question
    ###

    correlation = df_songs["popularity"].corr(df_songs["avg_star_rating"])
    print("Correlation: ", correlation)

    ###
    # Using linear regression to answer the question.
    ###

    X = df_songs[["popularity"]]
    y = df_songs["avg_star_rating"]
    regressor = LinearRegression()
    regressor.fit(X, y)

    # Get the coefficients of the linear regression
    slope = regressor.coef_[0]
    intercept = regressor.intercept_

    # Create a scatterplot
    sns.set(style="whitegrid")  # Set seaborn style
    plt.figure(figsize=(8, 6))  # Set the figure size
    sns.scatterplot(
        x="popularity",
        y="avg_star_rating",
        data=df_songs,
        hue="num_ratings_category",
        palette="coolwarm",
        legend="full",
        hue_order=["Q1", "Q2", "Q3", "Q4", "Q5"],
    )

    # Plot the regression line
    x_range = np.linspace(
        df_songs["popularity"].min(), df_songs["popularity"].max(), 100
    )
    y_pred = slope * x_range + intercept
    plt.plot(
        x_range,
        y_pred,
        color="green",
        label=f"Regression Line (y = {slope:.4f}x + {intercept:.4f})",
    )

    # Add a title and labels
    plt.title("Scatterplot with Linear Regression")
    plt.xlabel("Popularity")
    plt.ylabel("Average Star Rating")

    # Show the legend
    plt.legend()

    # Show the plot
    plt.savefig("output/q9_regression.png", dpi=300)

    X = sm.add_constant(X)
    regressor = sm.OLS(y, X).fit()

    # Perform a t-test for the slope parameter
    t_statistic = regressor.tvalues["popularity"]
    p_value = regressor.pvalues["popularity"]

    # Define the significance level (alpha)
    alpha = 0.05

    # Check if the p-value is less than alpha
    if p_value < alpha:
        print(
            f"The p-value ({p_value:.4f}) is less than alpha ({alpha}), so the slope is statistically significant."
        )
    else:
        print(
            f"The p-value ({p_value:.4f}) is greater than alpha ({alpha}), so the slope is not statistically significant."
        )

    # Print the t-statistic and p-value
    print(f"T-Statistic: {t_statistic:.4f}")
    print(f"P-Value: {p_value:.4f}")

    print(regressor.summary())

    #########
    # Which 10 songs are in the “greatest hits” (out of the 5k songs), on the basis of the popularity based model?
    #########

    top_10_rating = df_songs.sort_values(by="avg_star_rating", ascending=False).head(
        10
    )[
        [
            "songNumber",
            "artists",
            "album_name",
            "track_name",
            "popularity",
            "avg_star_rating",
            "num_ratings",
        ]
    ]
    joblib.dump(
        df_songs.sort_values(by="avg_star_rating", ascending=False).songNumber.values,
        "q9_avg_star_rating.pkl",
    )
    dfi.export(
        top_10_rating,
        "output/q9_top_10_rating.png",
        table_conversion="matplotlib",
        dpi=300,
    )

    top_10_popularity = df_songs.sort_values(by="popularity", ascending=False).head(10)[
        [
            "songNumber",
            "artists",
            "album_name",
            "track_name",
            "popularity",
            "avg_star_rating",
            "num_ratings",
        ]
    ]
    dfi.export(
        top_10_popularity,
        "output/q9_top_10_popularity.png",
        table_conversion="matplotlib",
        dpi=300,
    )

    print(
        f"Popularity vs rating changes the Top 10 by {len(set(top_10_rating.songNumber) - set(top_10_popularity.songNumber))} songs."
    )


@question
def q10():
    pass


if __name__ == "__main__":
    q3()
    q4()
    q5()
    q8()
    print(-1)