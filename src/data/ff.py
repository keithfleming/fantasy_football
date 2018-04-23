import pandas as pd
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
        X = data[['Cmp', 'PassAtt', 'PassYds', 'PassTD', 'Int', 'RushAtt', 'RushYds', 'RushTD', 'FantPt']]
    elif pos == 'WR' or pos == 'TE':
        X = data[['Rec', 'Tgt', 'RecYds', 'RecTD', 'FantPt']]
    elif pos == 'RB':
        X = data[['RushAtt', 'RushYds', 'RushTD', 'Rec', 'Tgt', 'RecYds', 'RecTD', 'FantPt']]

    result = result[result['Label'].isin(index)]
    result = result.sort_values('Label')
    Y = result['FantPt']

    #Fill in any missing values with 0
    X = X.fillna(0)
    Y = Y.fillna(0)

    return X,Y