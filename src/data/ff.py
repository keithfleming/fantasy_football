import pandas as pd
import numpy as np
from sklearn  import linear_model
from sklearn import metrics
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

def format_name(seasonData):
    """
    The name variable for each player has 3 parts: first name, last name, and unique identifier. Each name will need to
    be seperated into the three components
    :param seasonData: The dataframe with all the players from a season
    :return: Dataframe with the name comlumn properly seperated
    """
    pass

def norm_data(seasonData):
    """
    The season data needs to be normalized
    :param seasonData: The complete set of data for a season
    :return: The data frame with the data normalized
    """
    pass

def fill_zeros(seasonData):
    """
    Many players have blanks for some cells. This needs to be filled with a 0 instead of NaN
    :param seasonData: The complete set of data for a season
    :return: The season data with zeros instead of NaN
    """
    pass

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
    :param index:
    :param pos:
    :return:
    """
    data = data[data['Label'].isin(index)]
    data = data.sort_values('Label')

    if pos == 'QB':
        X = data[['RecYds', 'FantPt']]
    elif pos == 'WR':
        X = data[['RecYds', 'FantPt']]
    elif pos == 'RB':
        X = data[['RecYds', 'FantPt']]
    elif pos == 'TE':
        X = data[['RecYds', 'FantPt']]

    result = result[result['Label'].isin(index)]
    result = result.sort_values('Label')
    Y = result['FantPt']

    X = X.fillna(0)
    Y = Y.fillna(0)

    return X,Y

if __name__ == "__main__":
    train = load_data("2013_Fantasy")
    test = load_data("2014_Fantasy")
    QB_2013_2014_index = get_index_for_position(train,test)

    input, result = prepare_input_data(train, test,  QB_2013_2014_index, 'QB')
    QBmodel = linear_model.LinearRegression()
    QBmodel.fit(input,result)
    print(QBmodel.coef_)







