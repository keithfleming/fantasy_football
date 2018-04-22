import matplotlib.pyplot as plt
import numpy as np
from math import ceil

def plot_regression_coefs(multiple, ridge, lasso, elasticnet, pos):
    width = 0.2

    if pos == 'QB':
        labels = ['Cmp', 'PassAtt', 'PassYds', 'PassTD', 'Int', 'RushAtt', 'RushYds', 'Y/A', 'RushTD', 'FantPt']
    elif pos == 'RB':
        labels = []
    else:
        labels = []

    r1 = np.arange(len(multiple))
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    r4 = [x + width for x in r3]

    plt.bar(r1, multiple, color = 'r', width = width, label = 'Linear')
    plt.bar(r2, ridge, color = 'b', width = width, label = 'Ridge')
    plt.bar(r3, lasso, color='g', width=width, label='Lasso')
    plt.bar(r4, elasticnet, color='y', width=width, label='Elastic Net')

    plt.style.use('seaborn')
    plt.xticks(rotation=45)
    plt.title(pos + ' Regression Model Coefficients')
    plt.xlabel('Coefficient')
    plt.ylabel('Coefficient Value')
    plt.xticks([r + width for r in range(len(multiple))], labels)

    plt.legend()
    plt.show()

def plot_pred_vs_actual(pred, actual, pos, model):
    extrema = max(max(pred), max(actual))
    extrema = int(ceil(extrema / 100.0)) * 100

    plt.plot([0,extrema], [0,extrema], '--', color = 'b')
    plt.scatter(pred, actual)
    plt.axis('equal')

    plt.ylim(0,extrema)
    plt.xlim(0,extrema)

    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title(pos + ' ' + model + ' Model Predicted vs. Actual')

    plt.show()

