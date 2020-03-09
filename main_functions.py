
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def outliers_with_IQR(data,key,kernal=1.5):
    import statistics
    med = int(len(data[key])/2)
    sorted_data = sorted(data[key])
    Q1 = statistics.median(sorted_data[:med])
    Q2 = statistics.median(sorted_data[med:])
    iqr = Q2 - Q1
    outliers_range_1 = Q1 - iqr*kernal
    outliers_range_2 = Q2 + iqr*kernal
    data = data[data[key]>outliers_range_1]
    data = data[data[key]<outliers_range_2]
    return data


def hypothesis(model):
    try:
        print(type(model).__name__,' Model coefficient ',model.coef_)
        print(type(model).__name__,' Model collinearity ',model.intercept_)
    except:
        print(type(model).__name__,' Has no coefficient and collinearity')

def score(y_pred,y_test):
    from sklearn.metrics import mean_squared_error , r2_score
    print('Mean Squared Error : ', mean_squared_error(y_pred,y_test))
    print('R______Score Error : ', r2_score(y_pred,y_test))

def display(y_test,y_pred):
    plt.figure(figsize=(10,8))
    plt.title('PREDICTION VS ACTUAL ', fontsize=24 , fontstyle='italic')
    plt.plot(range(len(y_test)), np.sort(y_test), '.')
    plt.plot(range(len(y_test)),np.sort(y_pred),c='r')
    plt.show()