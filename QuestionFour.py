import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from scipy.stats import yeojohnson
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import re

np.random.seed(13427256)


def addStrings(file):
    with open(file, "r") as texttoList:
        colList = re.split(', | /s | and | or', texttoList.read().strip())
    return colList


def plotBoxPlot(data, ax, featureName, desc=""):
    ax.boxplot(data)
    ax.set_title(f"Boxplot of {featureName} {desc}")
    # ax.set(xlabel='x-label', ylabel='y-label')


def plotResiduals(yPred, ytest, ax, desc=""):
    residuals = ytest - yPred
    ax.scatter(yPred, residuals)
    ax.set_title(f"Scatter plot of residuals {desc}")
    # ax.set(xlabel='predicted values', ylabel='residuals')


def data_prep():
    #colList = addStrings(columnNameFile)
    data = pd.read_csv("spotify52kData.csv")
    datagpd = (data.groupby(["track_name", "artists"]).max().reset_index().sort_values(by="songNumber")
               .reset_index(drop=True))  # Groups identical songs on different albums by the same artist into 1
    # group
    datagpd["duration"] = datagpd["duration"] / (1000 * 60)  # convert duration to minutes from milliseconds
    return datagpd


def findSplitLimit(data, featureName):
    resultList = boxplot_stats(data[featureName])[0]
    upperwhisker, lowerwhisker = resultList["whishi"], resultList["whislo"]
    return upperwhisker, lowerwhisker


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


def Linear_regression_with_datasplit(data, featureName):
    upper, lower = findSplitLimit(data, featureName)
    featureData = data[featureName]
    condition = (featureData >= lower) & (featureData <= upper)
    dataNormal = data[condition]
    dataFlier = data[~condition]
    ypredNormal, ytestNormal = Linear_regression(dataNormal[featureName], dataNormal["popularity"], transform=False,
                                                 desc="(for values between whiskers")
    ypredflier, ytestflier = Linear_regression(dataFlier[featureName], dataFlier["popularity"], transform=False,
                                               desc="(for flier values)")
    return ypredNormal, ytestNormal, ypredflier, ytestflier


def transform_split_residual_comparison(data, predictors, predictorsNotSplit):
    for predictor in predictors:
        fig, axfeat = plt.subplots(1, 2)
        fig2, axresiduals = plt.subplots(2, 2)
        fig.suptitle(f"Boxplots of {predictor}")
        fig2.suptitle(f"Scatter plot of residuals ({predictor} vs Popularity)")
        fig2.supxlabel("predicted values (ypred)")
        fig2.supylabel("Residuals (y true - y predicted")

        plotBoxPlot(data[predictor], axfeat[0], predictor)
        featureTransformed = yeojohnson(data[predictor])[0]
        plotBoxPlot(featureTransformed, axfeat[1], predictor, desc="(transformed)")
        ypred, ytest = Linear_regression(data[predictor], data["popularity"], transform=False, desc="(As is)")
        ypredTrfm, _ = Linear_regression(data[predictor], data["popularity"], transform=True, desc="(Transformed)")
        plotResiduals(ypred, ytest, axresiduals[0][0])
        plotResiduals(ypredTrfm, ytest, axresiduals[0][1], desc="(Transformed (yeo - johnson)")
        if predictor in predictorsNotSplit:
            continue
        else:
            ypredNormal, ytestNormal, ypredflier, ytestflier = Linear_regression_with_datasplit(data, predictor)
            plotResiduals(ypredNormal, ytestNormal, axresiduals[1][0], desc="(for values between whiskers)")
            plotResiduals(ypredflier, ytestflier, axresiduals[1][1], desc="(for flier (higher than outliers) values)")
        plt.show()


if __name__ == "__main__":
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
