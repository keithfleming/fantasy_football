import pandas as pd
import numpy as np
from sklearn  import linear_model
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt

plt.style.use('ggplot')

def load_data(filename):
    """
    Function to load the raw data
    :param filename: the filename with the data to be loaded
    :return: dataframe with the loaded data
    """
    filename = "C:/Users/Keith/Documents/EECE2300/Homework_1/python/fantasy_football/data/raw/" + filename + ".csv"
    df = pd.read_csv(filename, header = [1])
    df = df.rename(index=str, columns={"Att": "PassAtt", "TD": "PassTD", "Yds": "PassYds", "Att.1": "RushAtt",
                                       "Yds.1": "RushYds", "TD.1": "RushTD", "Yds.2": "RecYds", "TD.2": "RecTD"})
    df = remove_players_wo_positions(df)
    df = format_position(df)
    df['Label'] = df['Name'].str.split('\\').str[1]
    df['Name'] = df['Name'].str.split('\\').str[0]
    return pd.DataFrame(df)

def remove_players_wo_positions(df):
    """
    Remove all players without a listed position
    :param df:
    :return: The predicted player position will be returned
    """
    df = df[pd.notnull(df['FantPos'])]
    return df

def format_position(df):
    """
    The player position has different formatting which will all need to be capatialized for grouping
    Alternatively, could remove players without positions because they do not do much
    :param player: The player to fix formatting for
    :return: The reformatted player position
    """
    df['FantPos'] = df['FantPos'].str.upper()
    return df

def plot_yards_by_position(df):
    """
    Generate a plot of total average yards per position
    :param df: The dataframe with a season of data
    :return: None
    """
    positiondf = df.groupby('FantPos').mean()
    ax = positiondf[['PassYds', 'RushYds', 'RecYds']].plot(kind = 'bar', stacked = True, title = 'Total Yards by Position')
    ax.set_xlabel('Position')
    ax.set_ylabel('Yards')

def hist_yards_by_pos(df, pos = 'QB'):
    """
    Generate a histogram with yards for a specific position
    :param df: The dataframe with a season of data
    :param pos: The position to create the histogram for
    :return: None
    """
    df = df[df['FantPos'] == pos]
    ax = df[['PassYds', 'RushYds', 'RecYds']].plot.hist(bins = 25, stacked = True, title = 'Histogram of Yardage for ' + pos)
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Yards')


def total_fantasy_points_by_team(df):
    """
    Generate a plot with the average number of fantasy points per team
    :param df: The dataframe with a season of data
    :return: None
    """
    df = df.loc[:,'Tm':'FantPt']
    df = df.drop('Age',1)
    df = df.drop('FantPos', 1)
    teamdf = df.groupby('Tm').sum()
    teamdf = teamdf.drop(['2TM', '3TM'])
    ax = teamdf['FantPt'].sort_values(ascending = False).plot.bar(title = 'Total Fantasy Points by Team')
    ax.set_xlabel('Team')
    ax.set_ylabel('Fantasy Points')

def get_index_for_position(season1, season2, pos = 'QB'):
    """
    Function that returns the index of common players for a specific position
    :param season1: The first year of season data
    :param season2: The send year of season data
    :param pos: The position to get the index for
    :return: The index with unique players
    """
    pos1 = season1[season1['FantPos'] == pos]
    pos2 = season2[season2['FantPos'] == pos]
    lab1 = pos1['Label'].tolist()
    lab2 = pos2['Label'].tolist()
    player_index = [x for x in lab1 if x in lab2]
    return player_index

def prepare_input_data(data, result, index, pos):
    """
    Prepare the training and test data for modeling
    :param data:
    :param result:
    :param index: The set of players to prepare data for
    :param pos: The position to aggregrate data for
    :return:
    """
    data = data[data['Label'].isin(index)]
    data = data.sort_values('Label')

    if pos == 'QB':
        X = data[['Cmp', 'PassAtt', 'PassYds', 'PassTD', 'Int', 'RushAtt', 'RushYds', 'Y/A', 'RushTD', 'FantPt']]
    elif pos == 'WR':
        X = data[['RecYds', 'FantPt']]
    elif pos == 'RB':
        X = data[['RecYds', 'FantPt']]
    elif pos == 'TE':
        X = data[['RecYds', 'FantPt']]

    result = result[result['Label'].isin(index)]
    result = result.sort_values('Label')
    Y = result['FantPt']

    #Fill in any missing values with 0
    X = X.fillna(0)
    Y = Y.fillna(0)

    return X,Y

def linear_regression(training_stats, training_future_points, test_stats, test_future_points):
    """
        Fit and test a linear regression model
        :param training_stats: The dataframe with player statistics to fit the model
        :param training_future_points: The set of future fantasy points for the training_stats
        :param test_stats: The dataframe with player statistics to test model
        :param test_future_points: The set of future fantasy points for the test_stats
        :return: Model intercept, model coefficients, and mse from the test set
        """
    model = linear_model.LinearRegression()
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
    model = linear_model.RidgeCV()
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
    model = linear_model.LassoCV()
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
        scores = cross_val_predict(knn, training_stats, training_future_points, cv = 5)
        cv_scores.append(scores.mean())

    optimal_k = neighbors[cv_scores.index(min(cv_scores))]

    model = KNeighborsRegressor(n_neighbors=optimal_k)
    model.fit(training_stats, training_future_points)

    pred = model.predict(test_stats)

    mse = metrics.mean_squared_error(test_future_points, pred)

    return model, pred, mse

if __name__ == "__main__":
    fantasy2013 = load_data("2013_Fantasy")
    fantasy2014 = load_data("2014_Fantasy")
    fantasy2015 = load_data("2015_Fantasy")

    QB_2013_2014_index = get_index_for_position(fantasy2013,fantasy2014)
    QB_2014_2015_index = get_index_for_position(fantasy2014, fantasy2015)

    QBstats2013, QBpoints2014 = prepare_input_data(fantasy2013, fantasy2014,  QB_2013_2014_index, 'QB')
    QBstats2014, QBpoints2015 = prepare_input_data(fantasy2014, fantasy2015, QB_2014_2015_index, 'QB')

    QB_linear_model, QB_linear_preds, QB_linear_mse = linear_regression(QBstats2013, QBpoints2014, QBstats2014, QBpoints2015)
    QB_ridge_model, QB_ridge_preds, QB_ridge_mse = ridge_regression(QBstats2013, QBpoints2014, QBstats2014, QBpoints2015)
    QB_lasso_model, QB_lasso_preds, QB_lasso_mse = ridge_regression(QBstats2013, QBpoints2014, QBstats2014, QBpoints2015)
    QB_knn_model, QB_knn_preds, QB_knn_mse = knn(QBstats2013, QBpoints2014, QBstats2014, QBpoints2015)

    #mse = metrics.mean_squared_error(points2015, pred2015)
    #mae = metrics.mean_absolute_error(points2015, pred2015)
    #resid = points2015 - pred2015

    #plt.hist(resid, 20)
    #plt.xlabel('Residuals')
    #plt.ylabel('Frequency')
    #plt.title('2015 QB Model Residuals')
    #plt.show()