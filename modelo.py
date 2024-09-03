import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def loadData():
    train = pd.read_csv("train_dataset.csv")
    test = pd.read_csv("test_dataset.csv")
    return (train, test)


def splitData(data):
    x = data.drop(["target"], axis=1)
    y = data["target"]
    return (x, y)


def modeloRegresion(xTrain, yTrain):
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(xTrain, yTrain)

    return model


def eval_metrics(y_true, y_pred):

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return mse, mae, r2


def saveModel(model):
    import pickle

    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)

    return None


def main():
    train, test = loadData()
    xTrain, yTrain = splitData(train)
    xTest, yTest = splitData(test)
    modelo = modeloRegresion(xTrain, yTrain)
    saveModel(modelo)

    yTrainPred = modelo.predict(xTrain)
    yTestPred = modelo.predict(xTest)

    print(eval_metrics(yTrain, yTrainPred))
    print(eval_metrics(yTest, yTestPred))


main()
