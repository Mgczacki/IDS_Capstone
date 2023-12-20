# Our random seed
seed = 13427256

import joblib
import seaborn as sns
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
import dataframe_image as dfi

from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from typing import Callable


def question(q: Callable):
    """Wrapper function for questions in order to initialize them.

    Args:
        q (function): Function that answers a question in the Capstone Project.
    """
    def wrapped_question():
        print(f"##### Starting {q.__name__.upper()} #####")
        np.random.seed(seed)
        q()

    return wrapped_question


@question
def q1():
    pass


@question
def q2():
    pass


@question
def q3():
    pass


@question
def q4():
    pass


@question
def q5():
    pass


@question
def q6():
    pass


@question
def q7():
    pass


@question
def q8():
    pass


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


q1(), q2(), q3(), q4(), q5(), q6(), q7(), q8(), q9(), q10()
