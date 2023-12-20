import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import yeojohnson
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import train_test_split
from QuestionFour import plotResiduals, rms_and_r2, data_prep


def multi_regression(X, y, transform=True, desc=""):
    y = y.to_numpy().reshape(-1, 1)
    X = X.to_numpy()
    if transform:
        for col in range(X.shape[1]):
            X[:, col] = yeojohnson(X[:, col])[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # model = Ridge(alpha=0.0001)
    model = RidgeCV(alphas=np.logspace(-6, 6, 13))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmseErr, r2 = rms_and_r2(y_pred, y_test)
    print(f"RMSE error {desc} is {rmseErr}, R2: {r2}\n")
    return y_pred, y_test


if __name__ == "__main__":
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
