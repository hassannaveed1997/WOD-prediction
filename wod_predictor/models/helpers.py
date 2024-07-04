import pandas as pd
from IPython.display import display

def show_breakdown_by_workout(y_pred, y_test):
    y_pred_means = y_pred.mean(axis=0)
    y_test_means = y_test.mean(axis=0)

    df = pd.DataFrame({'y_test_mean':y_test_means, 'y_pred_mean':y_pred_means})
    df['error'] = (df['y_test_mean'] - df['y_pred_mean'])
    df['error_percentage'] = df['error'] / df['y_test_mean']*100

    display(df)
    
