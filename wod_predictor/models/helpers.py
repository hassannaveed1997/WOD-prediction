import pandas as pd
from IPython.display import display
from sklearn.metrics import mean_absolute_error

def show_breakdown_by_workout(y_pred, y_test):
    y_pred_means = y_pred.mean(axis=0)
    y_test_means = y_test.mean(axis=0)
    error_df = (y_test - y_pred).abs()

    error_means = error_df.mean(axis=0)

    df = pd.DataFrame(
        {
            "y_test_mean": y_test_means,
            "y_pred_mean": y_pred_means,
            "error_mean": error_means,
        }
    )
    df["error_percentage"] = df["error_mean"] / df["y_test_mean"] * 100

    display(df)

def show_comparison(target, benchmark, predictions):
    metrics = {
        'MAE':mean_absolute_error
    }
    results = []
    for name, metric in metrics.items():
        metric_results = pd.DataFrame({
            'benchmark': metric(y_true = target, y_pred = benchmark),
            'model': metric(y_true = target, y_pred = predictions),
        }, index=[name])

        results.append(metric_results)
    results_df =  pd.concat(results, axis = 0)

    results_df["improvement"] = (results_df['benchmark']-results_df['model'])/results_df['benchmark']
    print("\nModel Comparison:")
    display(results_df)

def unstack_series(series, meta_data):
    """
    Unstack a series with a multiindex
    """
    df = pd.DataFrame(series)
    df["workout_name"] = df.index.map(meta_data["idx_to_workout_name"])
    df["athlete_id"] = df.index.map(meta_data["idx_to_athlete_id"])
    return df.pivot(columns="workout_name", values="score", index="athlete_id")
