# Our random seed
seed = 13427256

import gc
import math
from typing import Callable

import dataframe_image as dfi
import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
import statsmodels.api as smf
from mpl_toolkits import mplot3d
from scipy.stats import mannwhitneyu, norm, permutation_test, yeojohnson
from sklearn import metrics
from sklearn.cluster import DBSCAN as sk_DBSCAN
from sklearn.cluster import KMeans as sk_KMeans
from sklearn.decomposition import PCA as sk_PCA
from sklearn.linear_model import (Lasso, LinearRegression, LogisticRegression,
                                  RidgeCV)
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             mean_squared_error, r2_score, silhouette_samples,
                             silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

question_list = []


def question(q: Callable):
    """Wrapper function for questions in order to initialize them.

    Args:
        q (function): Function that answers a question in the Capstone Project.
    """

    def wrapped_question():
        print("########################")
        print(f"##### Starting {q.__name__.upper()} #####")
        print("########################")
        np.random.seed(seed)
        q()
        gc.collect()

    question_list.append(wrapped_question)

    return wrapped_question


##################################
# Data loading and preprocessing #
##################################

df_songs = pd.read_csv("spotify52kData.csv").sort_values(
    by="popularity", ascending=False
)
df_songs_uncleaned = df_songs.copy()

indices_of_duplicates = df_songs[
    df_songs.duplicated(keep="first", subset=["artists", "track_name"])
].index.tolist()
df_songs = df_songs.drop(indices_of_duplicates).sort_values(by="songNumber")
df_songs["duration"] = df_songs["duration"] / (
    1000 * 60
)  # convert duration to minutes from milliseconds
# For Q1
df_songs = df_songs.assign(logduration=np.log(df_songs[["duration"]]))
df_songs_original = df_songs.copy()

df_songs_5k = df_songs.query("songNumber < 5000")
df_stars = pd.read_parquet("starRatings.parquet").drop(
    columns=[x for x in indices_of_duplicates if x < 5000]
)

#############
# Questions #
#############


@question
def q1():
    """Is there a relationship between song length and popularity of a song? If so, is it positive or negative?

    Author: Yuan Huang
    """
    # Using correlation
    correlation = df_songs["popularity"].corr(df_songs["duration"])
    print("Correlation: ", correlation)

    # Using regression
    X = df_songs[["duration"]]
    y = df_songs["popularity"]
    regressor = LinearRegression()
    regressor.fit(X, y)

    slope = regressor.coef_[0]
    intercept = regressor.intercept_

    sns.set(style="whitegrid")  # Set seaborn style
    plt.figure(figsize=(8, 6))  # Set the figure size
    sns.scatterplot(
        x="duration", y="popularity", data=df_songs, palette="coolwarm", legend="full"
    )

    # Plot the regression line
    x_range = np.linspace(df_songs["duration"].min(), df_songs["duration"].max(), 100)
    y_pred = slope * x_range + intercept
    plt.plot(
        x_range,
        y_pred,
        color="green",
        label=f"Regression Line (y = {slope:.6f}x + {intercept:.4f})",
    )

    # Add a title and labels
    plt.title("Scatterplot with Linear Regression of popularity on duration")
    plt.xlabel("duration")
    plt.ylabel("Popularity")

    # Show the legend
    plt.legend()

    # Save the plot
    plt.savefig("output/q1Scatterplot.png", dpi=300)

    X = sm.add_constant(X)
    regressor = sm.OLS(y, X).fit()

    # Perform a t-test for the slope parameter
    t_statistic = regressor.tvalues["duration"]
    p_value = regressor.pvalues["duration"]

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

    ### USING LOG TRANSFORMATION

    print("#### Now trying with a log transformation of the duration variable.")

    correlation = df_songs["popularity"].corr(df_songs["logduration"])

    X = np.log(df_songs[["duration"]])
    y = df_songs["popularity"]
    regressor = LinearRegression()
    regressor.fit(X, y)

    slope = regressor.coef_[0]
    intercept = regressor.intercept_

    sns.set(style="whitegrid")  # Set seaborn style
    plt.figure(figsize=(8, 6))  # Set the figure size
    sns.scatterplot(
        x="logduration",
        y="popularity",
        data=df_songs,
        palette="coolwarm",
        legend="full",
    )

    # Plot the regression line
    x_range = np.linspace(
        df_songs["logduration"].min(), df_songs["logduration"].max(), 100
    )
    y_pred = slope * x_range + intercept
    plt.plot(
        x_range,
        y_pred,
        color="green",
        label=f"Regression Line (y = {slope:.4f}x + {intercept:.4f})",
    )

    # Add a title and labels
    plt.title(
        "Scatterplot with Linear Regression of popularity on natural log duration"
    )
    plt.xlabel("natural log duration")
    plt.ylabel("Popularity")

    # Show the legend
    plt.legend()

    # Saving the plot
    plt.savefig("output/q1Scatterplot_log.png", dpi=300)

    X = sm.add_constant(X)
    regressor = sm.OLS(y, X).fit()

    # Perform a t-test for the slope parameter
    t_statistic = regressor.tvalues["duration"]
    p_value = regressor.pvalues["duration"]

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


@question
def q2():
    """Are explicitly rated songs more popular than songs that are not explicit?

    Author: Yuan Huang
    """

    non_explicit = df_songs[df_songs["explicit"] == False][["popularity"]]
    explicit = df_songs[df_songs["explicit"] == True][["popularity"]]

    plt.figure(figsize=(8, 6))
    # set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above)
    sns.set(style="darkgrid")

    sns.histplot(
        data=non_explicit, color="skyblue", label="Non-explicit songs", kde=True
    )
    sns.histplot(data=explicit, color="red", label="Explicit songs", kde=True)

    plt.xlabel("Popularity")
    plt.ylabel("Counts")
    plt.title(
        "Histogram of the popularity for the non-explicit songs and the explicit songs"
    )
    plt.legend()
    plt.savefig("output/q2Histogram.png", dpi=300)

    # difference in means
    print("Difference in means: ", np.mean(non_explicit) - np.mean(explicit))

    # difference in medians
    print("Difference in medians: ", np.median(non_explicit) - np.median(explicit))

    print(stats.mannwhitneyu(non_explicit, explicit))

    print(stats.ttest_ind(non_explicit, explicit))

    def statistic(x, y, axis=-1):
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    res = permutation_test(
        (non_explicit, explicit), statistic, alternative="less", n_resamples=1000
    )
    plt.figure(figsize=(8, 6))
    plt.hist(res.null_distribution, bins=100)
    plt.axvline(x=3, color="r")
    plt.xlabel("Difference of means")
    plt.ylabel("counts")
    plt.title(
        "Null distribution if non-explicit songs and the explicit songs have the same mean"
    )

    plt.savefig("output/q2PermTest.png", dpi=300)

    print("P-value of the permutation test: ", res.pvalue)


@question
def q3():
    """Are songs in major key more popular than songs in minor key?

    Author: Haris Naveed
    """
    q3data = df_songs[["mode", "popularity"]]

    MajorPop = pd.value_counts(
        q3data[q3data["mode"] == 1].popularity.values, normalize=True
    )
    MinorPop = pd.value_counts(
        q3data[q3data["mode"] == 0].popularity.values, normalize=True
    )
    fig = plt.figure()
    ax = plt.subplot()
    fig.suptitle("Popularity barplot for Major & Minor key songs")
    ax.bar(x=MajorPop.index, height=MajorPop, color="blue", label="Major")

    fig.supylabel("Probability Density")
    fig.supxlabel("Popularity Measure (Higher = More popular)")

    ax.bar(x=MinorPop.index, height=MinorPop, color="orange", label="Minor", alpha=0.8)
    fig.legend(frameon=False)

    testStat, pvalue = mannwhitneyu(MinorPop, MajorPop)
    print(
        f"pvalue ({pvalue}) is greater than our significance level 0.05, hence the null hypothesis (songs in either key"
        f"are equall fails to be rejected."
    )
    plt.savefig("output/q3ProbDensity.png", dpi=300)


@question
def q4():
    """Which of the following 10 song features: duration, danceability, energy, loudness, speechiness,
    acousticness, instrumentalness, liveness, valence and tempo predicts popularity best?
    How good is this model?

    Author: Haris Naveed
    """
    RMSE = dict()
    R2 = dict()

    predictors = [
        "duration",
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
    ]
    # predictors_to_not_split = ["danceability", 'energy', 'acousticness', 'tempo', 'valence']
    # transform_split_residual_comparison(data, predictors, predictors_to_not_split)
    for predictor in predictors:
        yPred, yTest = Linear_regression(
            df_songs[predictor],
            df_songs["popularity"],
            transform=True,
            desc="(Transformed)",
        )
        RMSE[predictor], R2[predictor] = rms_and_r2(yPred, yTest)

    resultsTable = (
        pd.DataFrame(RMSE.items(), columns=["Feature", "RMSE"])
        .sort_values(by="RMSE", ascending=False)
        .reset_index(drop=True)
    )
    resultsTable["R2 Score"] = [r2 for r2 in R2.values()]

    dfi.export(
        resultsTable,
        "output/q4_results.png",
        table_conversion="matplotlib",
        dpi=300,
    )


@question
def q5():
    """Building a model that uses *all* of the song features mentioned in question 4, how well can you predict popularity?
    How much (if at all) is this model improved compared to the model in question 4). How do you account for this? What happens
    if you regularize your model?

    Author: Haris Naveed
    """
    fig = plt.figure()
    fig.supxlabel("predicted values")
    fig.supylabel("Residuals")
    ax = plt.subplot()
    predictors = [
        "duration",
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
    ]

    yPred, yTest = multi_regression(
        df_songs[predictors], df_songs["popularity"], transform=True
    )
    plotResiduals(yPred, yTest, ax, desc="(for all 10 features)")
    plt.savefig("output/q5_residuals.png", dpi=300)


@question
def q6():
    """When considering the 10 song features in the previous question, how many meaningful principal components
    can you extract? What proportion of the variance do these principal components account for? Using these
    principal components, how many clusters can you identify? Do these clusters reasonably correspond to the
    genre labels in column 20 of the data?

    Author: Yuan Huang
    """
    songs_feature = df_songs_uncleaned[
        [
            "duration",
            "danceability",
            "energy",
            "loudness",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
        ]
    ]

    class PCA:
        """A method for doing dimensionality reduction by transforming the feature
        space to a lower dimensionality, removing correlation between features and
        maximizing the variance along each feature axis.
        """

        def _init__(self):
            self.eigenValues = None
            self.components = None

        def transform(self, X, n_components):
            """Fit the dataset to the number of principal components specified in the
            constructor and return the transformed dataset"""
            covariance_matrix = self.calculate_covariance_matrix(X)

            # Where (eigenvector[:,0] corresponds to eigenvalue[0])
            eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

            # Sort the eigenvalues and corresponding eigenvectors from largest
            # to smallest eigenvalue and select the first n_components
            idx = eigenvalues.argsort()[::-1]  # [3, 2, 1] ---> [2, 1, 0] --> [0, 1, 2]
            eigenvalues = eigenvalues[idx][:n_components]
            eigenvectors = np.atleast_1d(eigenvectors[:, idx])[:, :n_components]

            # Set the object variables
            self.eigenValues = eigenvalues
            self.components = eigenvectors

            # Project the data onto principal components
            X_transformed = X.dot(eigenvectors)

            return X_transformed

        def calculate_covariance_matrix(self, X, Y=None):
            """Calculate the covariance matrix for the dataset X"""
            if Y is None:
                Y = X
            n_samples = np.shape(X)[0]
            covariance_matrix = (1 / (n_samples - 1)) * (X - X.mean(axis=0)).T.dot(
                Y - Y.mean(axis=0)
            )

            return np.array(covariance_matrix, dtype=float)

    pca_pipeline = Pipeline([("scaling", StandardScaler()), ("pca", sk_PCA())])
    pca_pipeline.fit(songs_feature)

    eigVals = pca_pipeline[1].explained_variance_

    ## Kaiser Criterion: Consider all principal components with eigen values greater than 1.0
    eigVals = pca_pipeline[1].explained_variance_
    nComponents = 10
    x = np.linspace(1, nComponents, nComponents)
    plt.figure(figsize=(10, 8))
    plt.bar(x, eigVals, color="gray")
    plt.plot(
        [0, nComponents], [1, 1], color="orange"
    )  # Orange Kaiser criterion line for the fox
    plt.xlabel("Principal component")
    plt.ylabel("Eigenvalue")
    plt.title("Engenvalues of different components")

    plt.savefig("output/q6Kaiser.png", dpi=300)

    print(
        "According to Kaiser Criterion we should consider 6 principal components, so the data after dimensionality reduction should be (52000,3)"
    )

    covarExplained = eigVals / sum(eigVals) * 100
    print("Variance explained by the 3 PCs above is: %.3f " % (sum(covarExplained[:3])))

    ## Do PCA and plot the features. Since there are three types of iris, ideally we should be able to see 3 well seperated clusters
    pca = PCA()
    feature_transformed = pca.transform(
        StandardScaler().fit_transform(songs_feature), 3
    )

    df_feature_transformed = pd.DataFrame(
        feature_transformed, columns=["PC1", "PC2", "PC3"]
    )

    df_feature_transformed.to_csv("pca_features.csv", index=False)

    genre = df_songs_uncleaned[["track_genre"]]

    df_feature_transformed["genre"] = genre

    cats = {v: k for k, v in enumerate(df_feature_transformed["genre"].unique())}

    df_feature_transformed["num_genre"] = df_feature_transformed["genre"].apply(
        cats.get
    )

    # Creating figure
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    # Creating plot
    ax.scatter3D(
        df_feature_transformed[["PC1"]],
        df_feature_transformed[["PC2"]],
        df_feature_transformed[["PC3"]],
        alpha=0.7,
        c=df_feature_transformed[["num_genre"]],
    )
    plt.title(
        "3D scatter plot of songs' features with principal components with 52 genres"
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    ax.set_zlabel("PC3")
    # show plot
    plt.savefig("output/q6PCAplot", dpi=300)

    l = []
    for i in range(2, 53):
        km = sk_KMeans(
            n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=seed
        )
        km.fit_predict(feature_transformed)
        score = silhouette_score(feature_transformed, km.labels_, metric="euclidean")
        print("Silhouetter Score: %.3f" % score)
        l.append(score)

    fig = plt.figure(figsize=(10, 10))
    plt.plot(range(2, 53), l)
    plt.title("Silhouetter Score with different number of clusters")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouetter Score")

    plt.savefig("output/q6SilPlot", dpi=300)

    km = sk_KMeans(
        n_clusters=2, init="k-means++", max_iter=300, n_init=10, random_state=seed
    )
    cluster = km.fit_predict(feature_transformed)

    df_feature_transformed["kmeans"] = cluster

    # Creating figure
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")

    # Creating plot
    ax.scatter3D(
        df_feature_transformed[["PC1"]],
        df_feature_transformed[["PC2"]],
        df_feature_transformed[["PC3"]],
        alpha=0.5,
        c=df_feature_transformed[["kmeans"]],
    )
    plt.title(
        "3D scatter plot of songs' features with principal components with 2 Kmeans clusters"
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    ax.set_zlabel("PC3")

    # Save plot
    plt.savefig("output/q6plot3D.png", dpi=300)

    c1 = df_feature_transformed.query("kmeans == 0")["genre"].value_counts()
    c2 = df_feature_transformed.query("kmeans == 1")["genre"].value_counts()

    print(((c1 - c2).abs() / 1000).fillna(1).sort_values())


@question
def q7():
    """Can you predict whether a song is in major or minor key from valence using logistic regression or a
    support vector machine? If so, how good is this prediction? If not, is there a better one?

    Author: Yuan Huang
    """
    df_q7 = df_songs_original[["mode", "valence"]].astype(float)

    plt.scatter(df_q7["valence"].values, df_q7["mode"].values)
    plt.title("Scatterplot of mode on valence")
    plt.xlabel("valence")
    plt.ylabel("mode")
    plt.savefig("output/q7_scatter_mode_valence.png", dpi=300)

    sns.histplot(df_q7, x="valence", hue="mode")
    plt.title("Histogram of valence given mode")
    plt.savefig("output/q7_hist_mode_valence.png", dpi=300)

    def k_fold_cv_train(X_train, y_train, num_folds=5, test_size=0.2):
        """
        Performs a k-fold validation backed training loop. Returns the best model.
        """
        best_model = None
        best_score = 0

        # CV training
        for fold in range(num_folds):
            model = LogisticRegression(random_state=seed)

            # Split the data into training and validation sets for this fold
            X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
                X_train, y_train, test_size=test_size, random_state=fold
            )

            # Fit the model on the training data for this fold
            model.fit(X_train_fold.values.reshape(-1, 1), y_train_fold)

            # Calculate the accuracy on the validation set for this fold
            score = model.score(X_val_fold.values.reshape(-1, 1), y_val_fold)

            # Check if this fold has the best performance
            if score > best_score:
                best_score = score
                best_model = model

        return best_model

    X_train, X_test, y_train, y_test = train_test_split(
        df_q7[["valence"]], df_q7[["mode"]], test_size=0.25, random_state=seed
    )

    logreg = k_fold_cv_train(X_train, y_train)

    y_pred = logreg.predict(X_test.values.reshape(-1, 1))

    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

    print("Confusion matrix: ", cnf_matrix)

    y_pred_proba = logreg.predict_proba(X_test.values.reshape(-1, 1))[::, 1]
    print("beta_1: ", logreg.coef_, "beta_0: ", logreg.intercept_)
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label="predicted mode by valence, auc=" + str(auc))
    plt.legend(loc=4)
    plt.savefig("output/q7auc.png", dpi=300)

    probit = smf.Probit(y_train, X_train)
    probit.fit()
    print(probit.fit().summary())

    y_pred = probit.fit().predict(X_test)

    df_probit = pd.DataFrame({"y_pred": y_pred})

    df_probit["y_pred_convert"] = np.where(df_probit["y_pred"] >= 0.5, 1, 0)

    print("Accuracy: ", accuracy_score(y_test, df_probit["y_pred_convert"].values))
    print("F1: ", f1_score(y_test, df_probit["y_pred_convert"].values))


@question
def q8():
    """Can you predict genre by using the 10 song features from question 4 directly or the principal components
    you extracted in question 6 with a neural network? How well does this work?

    Author: Haris Naveed
    """
    from sklearn.preprocessing import LabelEncoder

    LabelEncoder = LabelEncoder()
    num_samples = int(1e3)
    num_epochs = 50
    learn_rate = 1e-3
    batch_size = 10

    target = "track_genre"
    predictors = [
        "duration",
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
    ]
    train_original_features(
        df_songs,
        predictors,
        target,
        LabelEncoder,
        num_samples,
        num_epochs,
        learn_rate,
        batch_size,
    )
    PCA = pd.read_csv("pca_features.csv")
    X = PCA.values
    y = LabelEncoder.fit_transform(df_songs["track_genre"].to_numpy())
    train_data, train_labels, val_data, val_labels = data_setup(X, y)
    train_pca_features(
        num_samples,
        num_epochs,
        learn_rate,
        batch_size,
        train_data,
        train_labels,
        val_data,
        val_labels,
    )

@question
def q9():
    """In recommender systems, the popularity based model is an important baseline. We have a two part question in this regard:
    a) Is there a relationship between popularity and average star rating for the 5k songs we have explicit feedback for?
    b) Which 10 songs are in the “greatest hits” (out of the 5k songs), on the basis of the popularity based model?

    Author: Mario Garrido
    """
    avg_star_rating = df_stars.mean(axis=0)
    num_ratings = df_stars.count(axis=0)
    df_songs_5k["avg_star_rating"] = avg_star_rating
    df_songs_5k["num_ratings"] = num_ratings
    df_songs_5k["num_ratings_category"] = pd.qcut(
        df_songs_5k["num_ratings"], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
    )

    #########
    # Is there a relationship between popularity and average star rating for the 5k songs we have explicit feedback for?
    #########

    ###
    # Using correlation to answer the question
    ###

    correlation = df_songs_5k["popularity"].corr(df_songs_5k["avg_star_rating"], method="spearman")
    print("Correlation: ", correlation)

    ###
    # Using linear regression to answer the question.
    ###

    X = df_songs_5k[["popularity"]]
    y = df_songs_5k["avg_star_rating"]
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
        data=df_songs_5k,
        hue="num_ratings_category",
        palette="coolwarm",
        legend="full",
        hue_order=["Q1", "Q2", "Q3", "Q4", "Q5"],
    )

    # Plot the regression line
    x_range = np.linspace(
        df_songs_5k["popularity"].min(), df_songs_5k["popularity"].max(), 100
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

    top_10_rating = df_songs_5k.sort_values(by="avg_star_rating", ascending=False).head(
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
        df_songs_5k.sort_values(by="avg_star_rating", ascending=False).songNumber.values,
        "q9_avg_star_rating.pkl",
    )
    dfi.export(
        top_10_rating,
        "output/q9_top_10_rating.png",
        table_conversion="matplotlib",
        dpi=300,
    )

    top_10_popularity = df_songs_5k.sort_values(by="popularity", ascending=False).head(10)[
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
    """You want to create a “personal mixtape” for all 10k users we have explicit feedback for. This mixtape contains
    individualized recommendations as to which 10 songs (out of the 5k) a given user will enjoy most. How do these
    recommendations compare to the “greatest hits” from the previous question and how good is your recommender system
    in making recommendations?

    Author: Mario Garrido
    """

    def generate_user_features(user_number):
        """
        Generates a user's features. Assumes that the effect of the song to predict for is negligible next to the other songs.
        """

        features = {}
        features["average_rating"] = df_stars.loc[user_number].mean()

        user_weights = df_stars.loc[user_number].fillna(0)

        # Generate weighted numeric variables
        for col in [
            "popularity",
            "duration",
            "danceability",
            "energy",
            "loudness",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
        ]:
            features[f"{col}_weighted"] = np.average(
                df_songs_5k[col], weights=user_weights
            )

        usr_song_idx = list(df_stars.columns[~df_stars.loc[user_number].isna()])
        df_usr_songs = df_songs_5k.loc[usr_song_idx]

        features["explicit_percentage"] = df_usr_songs["explicit"].sum() / len(
            df_usr_songs
        )

        # Generate categorical variable percentage values
        for col in ["track_genre", "mode"]:
            for value in df_songs_5k[col].unique():
                features[f"{col}_{value}"] = (df_usr_songs[col] == value).sum() / len(
                    df_usr_songs
                )

        return features

    user_features = [generate_user_features(i) for i in tqdm(range(len(df_stars)))]
    df_features = pd.DataFrame(user_features)

    def generate_datasets(test_prop=0.1, eval_prop=0.2):
        # First reserve 10% of all users, in order to evaluate the model.
        all_user_ids = list(range(len(df_features)))

        # 90/10 split
        len_test_users = int(np.ceil(len(df_features) * test_prop))
        test_users = np.random.choice(all_user_ids, size=len_test_users, replace=False)
        non_test_users = [u for u in all_user_ids if u not in test_users]

        # Generate the per-song datasets
        song_train_idx = dict()
        song_val_idx = dict()

        for s in tqdm(df_songs_5k.songNumber):
            # 80/20 split over the users that rated the song, for users not in the test dataset.
            song_users = (df_stars[~df_stars[s].isna()]).index
            non_test_song_users = list(
                set(song_users).intersection(set(non_test_users))
            )

            len_song_val_users = int(np.ceil(len(non_test_song_users) * eval_prop))
            song_val_users = np.random.choice(
                non_test_song_users, size=len_song_val_users, replace=False
            )

            song_train_users = [
                u for u in non_test_song_users if u not in song_val_users
            ]

            song_train_idx[s] = song_train_users
            song_val_idx[s] = song_val_users

        return test_users, song_train_idx, song_val_idx

    try:
        # Try to load the existing datasets
        test_users = joblib.load("q10_test_users.pkl")
        song_train_idx = joblib.load("q10_song_train_idx.pkl")
        song_val_idx = joblib.load("q10_song_val_idx.pkl")
        print("Loaded existing datasets.")
    except:
        # If the file doesn't exist, create and save the models
        print("Datasets not found. Generating...")

        test_users, song_train_idx, song_val_idx = generate_datasets()
        # Save the datasets
        joblib.dump(test_users, "q10_test_users.pkl")
        joblib.dump(song_train_idx, "q10_song_train_idx.pkl")
        joblib.dump(song_val_idx, "q10_song_val_idx.pkl")
        print("Generated/saved datasets.")

    # Training loop

    sampler = optuna.samplers.TPESampler(
        seed=seed
    )  # Make the sampler behave in a deterministic way.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def get_sample_weights(y_true):
        return np.sqrt((y_true + 2) / 3)

    def weighted_mse(y_true, y_pred):
        """
        Custom loss function that gives more weight to errors where the actual value is higher.
        """
        # Calculate squared errors
        squared_errors = (y_true - y_pred) ** 2

        # Weight errors by actual values
        weighted_errors = squared_errors * get_sample_weights(y_true)

        # Return mean weighted error
        return np.mean(weighted_errors)

    def get_pipeline(**kwargs):
        return Pipeline([("scaler", StandardScaler()), ("model", Lasso(**kwargs))])

    def train_song_model(song_number):
        X_train = df_features.loc[song_train_idx[song_number]].values
        y_train = df_stars[song_number].loc[song_train_idx[song_number]].values

        X_val = df_features.loc[song_val_idx[song_number]].values
        y_val = df_stars[song_number].loc[song_val_idx[song_number]].values

        sample_weights = get_sample_weights(y_train)

        def objective(trial):
            params = dict()
            params["alpha"] = trial.suggest_float("alpha", 0.001, 3.0, log=True)
            pipeline = get_pipeline(**params)
            pipeline.fit(X_train, y_train, model__sample_weight=sample_weights)
            y_pred = pipeline.predict(X_val)
            wmse = weighted_mse(y_val, y_pred)
            return wmse

        # Create an Optuna study
        study = optuna.create_study(direction="minimize", sampler=sampler)

        # Optimize the Lasso hyperparameters
        study.optimize(objective, n_trials=150)

        # Train the model with the best hyperparameters
        pipeline = get_pipeline(**study.best_params)
        pipeline.fit(X_train, y_train, model__sample_weight=sample_weights)
        return pipeline

    songs = df_songs_5k.songNumber.values

    try:
        # Try to load the existing model dictionary
        song_models = joblib.load("q10_song_models.pkl")
        print("Loaded existing model dictionary.")
    except:
        # If the file doesn't exist, create and save the models
        print("Model dictionary not found. Training models...")

        song_models = {s: train_song_model(s) for s in tqdm(songs)}
        # Save the trained models
        joblib.dump(song_models, "q10_song_models.pkl")
        print("Trained models and saved model dictionary.")

    # Model evaluation

    def pred_user(usr_number):
        """Gives the real ranking (by rating), and the full prediction by preference of a single user."""
        X = df_features.loc[[usr_number]]
        y = df_stars.loc[usr_number].fillna(0)
        usr_ratings = df_stars.loc[usr_number]
        listened = usr_ratings.dropna().index.tolist()
        liked = usr_ratings[usr_ratings > 3].index.to_list()
        y_pred = [(s, song_models[s].predict(X.values)[0]) for s in songs]
        pred_order = [x[0] for x in sorted(y_pred, key=lambda x: x[1], reverse=True)]
        real_order = list(y.sort_values(ascending=False).index)
        return real_order, pred_order, listened, liked

    test_user_ratings = [pred_user(u) for u in tqdm(test_users)]

    # From Q9, the average star rating based ranking
    avg_star_rating_order = joblib.load("q9_avg_star_rating.pkl")

    # Will use Jaccard similarity to measure how similar the users' top 10 lists are when compared to the popularity-based model.

    def jaccard(set1, set2):
        set1 = set(set1)
        set2 = set(set2)
        # intersection of two sets
        intersection = len(set1.intersection(set2))
        # Unions of two sets
        union = len(set1.union(set2))

        return intersection / union

    # Using Jaccard similarity to check how similar our predictions are to the popularity model's
    similarities = [
        jaccard(
            [x for x in r[1] if x not in r[2]][:10],
            [x for x in avg_star_rating_order if x not in r[2]][:10],
        )
        for r in tqdm(test_user_ratings)
    ]

    data_sorted = sorted(similarities)

    # Calculating the cumulative frequency
    cumulative_frequency = np.linspace(0.0, 1.0, len(similarities))

    plt.figure()
    # Creating the plot
    sns.lineplot(x=data_sorted, y=cumulative_frequency)
    plt.xlabel("Jaccard Similarity Predicted 10/Top 10 Popularity")
    plt.ylabel("Cumulative Frequency")
    plt.title("Cumulative Frequency Curve")

    # Show the plot
    plt.savefig("output/q10similarity.png", dpi=300)

    print("Mean similarity: ", np.mean(similarities))
    print("Median similarity: ", np.mean(similarities))

    # Using average precision @k to evaluate
    def average_precision(ground_truth, prediction, k=100):
        hits = 0
        sum_precisions = 0
        prediction = prediction[:k]
        for i, p in enumerate(prediction):
            if p in ground_truth:
                hits += 1
                precision_at_i = hits / (i + 1)
                sum_precisions += precision_at_i

        return sum_precisions / k

    at_k = 10

    pred_average_precisions = []
    pop_average_precisions = []

    for real, pred, listened, liked in test_user_ratings:
        pred_average_precisions.append(average_precision(liked, pred, k=at_k))
        pop_average_precisions.append(
            average_precision(liked, avg_star_rating_order, k=at_k)
        )

    plt.figure()
    sns.histplot(pred_average_precisions)
    plt.xlabel(f"Mean Average Precision @{at_k}")
    plt.ylabel("frequency")
    plt.title(f"Histogram of Mean Average Precision @{at_k} (ML Model)")
    plt.savefig("output/q10histMAP_ML.png", dpi=300)

    print("Mean average precision (our model): ", np.mean(pred_average_precisions))
    print("Median average precision (our model): ", np.median(pred_average_precisions))

    plt.figure()
    sns.histplot(pop_average_precisions)
    plt.xlabel(f"Mean Average Precision @{at_k}")
    plt.ylabel("frequency")
    plt.title(f"Histogram of Mean Average Precision @{at_k} (Popularity Model)")
    plt.savefig("output/q10histMAP_pop.png", dpi=300)

    print(
        "Mean average precision (popularity model): ", np.mean(pop_average_precisions)
    )
    print(
        "Median average precision (popularity model): ",
        np.median(pop_average_precisions),
    )


####################
# Helper Functions #
####################


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
        self.weight = math.sqrt(1.0 / (out_features * in_features)) * np.random.randn(
            out_features, in_features
        )
        self.bias = np.zeros(out_features)
        self.gradWeight = None
        self.gradBias = None

    def forward(self, x):  # this is our linear unit
        self.output = np.dot(x, self.weight.transpose()) + np.repeat(
            self.bias.reshape([1, -1]), x.shape[0], axis=0
        )
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


def train_model(
    num_epochs,
    learn_rate,
    batch_size,
    model,
    criterion,
    train_data,
    train_labels,
    val_data,
    val_labels,
):
    n_train, n_val = len(train_data), len(val_data)
    train_loss = np.empty([num_epochs, int(n_train / batch_size)])
    val_loss = np.empty([num_epochs, int(n_val / batch_size)])

    for epoch in range(num_epochs):
        # Training loop
        for i in range(int(n_train / batch_size)):
            x = train_data[batch_size * i : batch_size * (i + 1)]
            y = train_labels[batch_size * i : batch_size * (i + 1)]
            y_pred = model.forward(x)
            train_loss[epoch, i] = criterion.forward(y_pred, y)
            grad0 = criterion.backward(y_pred, y)
            grad = model.backward(x, grad0)
            model.gradientStep(learn_rate)

            # Validation loop
        for j in range(int(n_val / batch_size)):
            x = val_data[batch_size * j : batch_size * (j + 1)]
            y = val_labels[batch_size * j : batch_size * (j + 1)]
            y_pred = model.forward(x)
            val_loss[epoch, j] = criterion.forward(y_pred, y)

        if (epoch + 1) % 10 == 0:
            print("Training epoch:", epoch + 1)

    # Plot output, if desired
    plt.figure()
    plt.plot(np.mean(train_loss, axis=1))
    plt.plot(np.mean(val_loss, axis=1))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "val"])
    plt.show()


def evaluate_model(model, val_data, val_labels, batch_size, num_samples):
    n_val = len(val_data)
    y_pred = np.empty(
        [int(n_val / batch_size), batch_size]
    )  # if n_val/batch_size is not an integer it doesn't work

    for i in range(int(n_val / batch_size)):
        x = val_data[batch_size * i : batch_size * (i + 1)]
        y = val_labels[batch_size * i : batch_size * (i + 1)]
        y_pred[i, :] = np.argmax(model.forward(x), axis=1)

    rand_index = np.random.randint(len(val_data), size=num_samples)
    model_accuracy = np.mean((y_pred.flatten()[rand_index] == val_labels[rand_index]))

    return model_accuracy


def train_original_features(
    data,
    predictors,
    target,
    LabelEncoder,
    num_samples,
    num_epochs,
    learn_rate,
    batch_size,
):
    X = data[predictors].to_numpy()
    y = LabelEncoder.fit_transform(data[target].to_numpy())
    train_data, train_labels, val_data, val_labels = data_setup(X, y)
    model = MLP(input_features=10, hidden_size_1=32, hidden_size_2=64)
    criterion = CrossEntropyCriterion()
    train_model(
        num_epochs,
        learn_rate,
        batch_size,
        model,
        criterion,
        train_data,
        train_labels,
        val_data,
        val_labels,
    )
    model_accuracy = evaluate_model(
        model, val_data, val_labels, batch_size, num_samples
    )
    print("Model accuracy:", model_accuracy)


def train_pca_features(
    num_samples,
    num_epochs,
    learn_rate,
    batch_size,
    train_data,
    train_labels,
    val_data,
    val_labels,
):
    model = MLP(input_features=3, hidden_size_1=32, hidden_size_2=64)
    criterion = CrossEntropyCriterion()
    train_model(
        num_epochs,
        learn_rate,
        batch_size,
        model,
        criterion,
        train_data,
        train_labels,
        val_data,
        val_labels,
    )
    model_accuracy = evaluate_model(
        model, val_data, val_labels, batch_size, num_samples
    )
    print("Model accuracy:", model_accuracy)


def data_setup(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_data = X_train
    train_labels = y_train
    val_data = np.delete(
        X_val, -1, axis=0
    )  # This is done to solve a bug in evaluate_model that causes index overflow
    val_labels = np.delete(y_val, -1, axis=0)
    return train_data, train_labels, val_data, val_labels


[q() for q in question_list]

