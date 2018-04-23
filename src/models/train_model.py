import pandas as pd
import numpy as np
from sklearn  import linear_model
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score

def linear_regression(training_stats, training_future_points, test_stats, test_future_points):
    """
        Fit and test a linear regression model
        :param training_stats: The dataframe with player statistics to fit the model
        :param training_future_points: The set of future fantasy points for the training_stats
        :param test_stats: The dataframe with player statistics to test model
        :param test_future_points: The set of future fantasy points for the test_stats
        :return: Model intercept, model coefficients, and mse from the test set
        """
    model = linear_model.LinearRegression(fit_intercept=False)
    model.fit(training_stats, training_future_points)

    pred = model.predict(test_stats)

    mse = metrics.mean_squared_error(test_future_points, pred)

    return model, pred, mse

def ridge_regression(training_stats, training_future_points, test_stats, test_future_points):
    """
        Fit and test a ridge regression model
        :param training_stats: The dataframe with player statistics to fit the model
        :param training_future_points: The set of future fantasy points for the training_stats
        :param test_stats: The dataframe with player statistics to test model
        :param test_future_points: The set of future fantasy points for the test_stats
        :return: Ridge regression model and MSE from the test set
        """
    model = linear_model.RidgeCV(fit_intercept=False)
    model.fit(training_stats, training_future_points)

    pred = model.predict(test_stats)

    mse = metrics.mean_squared_error(test_future_points, pred)

    return model, pred, mse

def lasso_regression(training_stats, training_future_points, test_stats, test_future_points):
    """
        Fit and test a lasso regression model
        :param training_stats: The dataframe with player statistics to fit the model
        :param training_future_points: The set of future fantasy points for the training_stats
        :param test_stats: The dataframe with player statistics to test model
        :param test_future_points: The set of future fantasy points for the test_stats
        :return: Lasso regression model and MSE from the test set
        """
    model = linear_model.LassoCV(fit_intercept=False)
    model.fit(training_stats, training_future_points)

    pred = model.predict(test_stats)

    mse = metrics.mean_squared_error(test_future_points, pred)

    return model, pred, mse

def elasticnet_regression(training_stats, training_future_points, test_stats, test_future_points):
    """
        Fit and test a elastic net regression model
        :param training_stats: The dataframe with player statistics to fit the model
        :param training_future_points: The set of future fantasy points for the training_stats
        :param test_stats: The dataframe with player statistics to test model
        :param test_future_points: The set of future fantasy points for the test_stats
        :return: Elastic Net regression model and MSE from the test set
        """
    model = linear_model.ElasticNetCV(fit_intercept=False)
    model.fit(training_stats, training_future_points)

    pred = model.predict(test_stats)

    mse = metrics.mean_squared_error(test_future_points, pred)

    return model, pred, mse

def knn(training_stats, training_future_points, test_stats, test_future_points):
    """
        Fit and test a k nearest neighbor model
        :param training_stats: The dataframe with player statistics to fit the model
        :param training_future_points: The set of future fantasy points for the training_stats
        :param test_stats: The dataframe with player statistics to test model
        :param test_future_points: The set of future fantasy points for the test_stats
        :return: k nearest neighbors and MSE from the test set
        """
    neighbors = [2,4,6,8,10,12,14,16,18,20]

    cv_scores = []

    for k in neighbors:
        knn = KNeighborsRegressor(n_neighbors = k)
        scores = cross_val_score(knn, training_stats, training_future_points, cv = 5, scoring = 'neg_mean_squared_error')
        cv_scores.append(-scores.mean())

    optimal_k = neighbors[cv_scores.index(min(cv_scores))]

    model = KNeighborsRegressor(n_neighbors=optimal_k)
    model.fit(training_stats, training_future_points)

    pred = model.predict(test_stats)

    mse = metrics.mean_squared_error(test_future_points, pred)

    return model, pred, mse