import matplotlib.pyplot as plt
import numpy as np
from math import ceil

def plot_yards_by_position(df):
    """
    Generate a plot of total average yards per position
    :param df: The dataframe with a season of data
    :return: None
    """
    plt.figure()
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
    plt.figure()
    df = df[df['FantPos'] == pos]
    ax = df[['PassYds', 'RushYds', 'RecYds']].plot.hist(bins = 25, stacked = True, title = 'Histogram of Yardage for ' + pos)
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Yards')

def plot_regression_coefs(multiple, ridge, lasso, elasticnet, pos):
    plt.figure()
    width = 0.2

    if pos == 'QB':
        labels = ['Cmp', 'PassAtt', 'PassYds', 'PassTD', 'Int', 'RushAtt', 'RushYds', 'RushTD', 'FantPt']
    elif pos == 'RB':
        labels = ['RushAtt', 'RushYds', 'RushTD', 'Rec', 'Tgt', 'RecYds', 'RecTD', 'FantPt']
    else:
        labels = ['Rec', 'Tgt', 'RecYds', 'RecTD', 'FantPt']

    r1 = np.arange(len(multiple))
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    r4 = [x + width for x in r3]

    plt.bar(r1, multiple, color = '#cc0000', width = width, label = 'Linear')
    plt.bar(r2, ridge, color = 'b', width = width, label = 'Ridge')
    plt.bar(r3, lasso, color='g', width=width, label='Lasso')
    plt.bar(r4, elasticnet, color='#ff5500', width=width, label='Elastic Net')

    plt.style.use('seaborn')
    plt.xticks(rotation=45)
    plt.title(pos + ' Regression Model Coefficients', size = 14)
    plt.xlabel('Coefficient', size = 12)
    plt.ylabel('Coefficient Value', size = 12)
    plt.xticks([r + width for r in range(len(multiple))], labels, size = 12)
    plt.tight_layout()

    legend = plt.legend(frameon=1, prop={'size': 12})
    frame = legend.get_frame()
    frame.set_color('white')
    plt.show()

def plot_pred_vs_actual(pred, actual, pos, model):
    plt.figure()
    extrema = max(max(pred), max(actual))
    extrema = int(ceil(extrema / 100.0)) * 100

    plt.plot([-20,extrema+20], [-20,extrema+20], '--', color = 'b')
    plt.scatter(pred, actual, color = 'r')
    plt.axis('equal')

    plt.ylim(0,extrema)
    plt.xlim(0,extrema)

    plt.xlabel('Predicted Values', size = 12)
    plt.ylabel('Actual Values', size = 12)
    plt.title(pos + ' ' + model + ' Model Predicted vs. Actual', size = 14)
    plt.tight_layout()

    plt.show()

