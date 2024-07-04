import pandas as pd
from IPython.display import display

def show_breakdown_by_workout(y_test, y_pred, mapping):
    workout_name = y_test.index.map(mapping)
    df = pd.DataFrame({'workout_name':workout_name, 'y_test':y_test, 'y_pred':y_pred})

    df['avg_errors'] = (y_test - y_pred).abs()
    df = df.groupby('workout_name').agg('mean').reset_index().rename({'y_test':'avg_y_test', 'y_pred':'avg_y_pred'}, axis=1)
    df['avg_error_percentage'] = df['avg_errors'] / df['avg_y_test']*100

    # only show unscaled
    df = df[~df['workout_name'].str.contains('scaled')]
    df = df[~df['workout_name'].str.contains('foundation')]

    display(df)
    
