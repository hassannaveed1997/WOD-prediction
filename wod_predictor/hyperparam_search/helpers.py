import warnings, wod_predictor

from wod_predictor.data_loader import DataLoader
from wod_predictor.splitter import DataSplitter
from wod_predictor.preprocessor import DataPreprocessor 
from wod_predictor import modeling
from wod_predictor.hyperparam_search.tuner import HyperparamTuner, ParamRange

def fit_model(
    model_name,
    train_X,
    train_y,
    init_args=None,
    fit_args=None
):
    if init_args is None:
        init_args = {}
    if fit_args is None:
        fit_args = {}
    
    model = getattr(modeling, model_name)(**init_args)
    model.fit(train_X, train_y, **fit_args)
    return model

def fit_preprocessor(
    train_data,
    preprocessor_args=None
):
    if preprocessor_args is None:
        preprocessor_args = {}

    preprocessor = DataPreprocessor(config=preprocessor_args)
    preprocessor.fit(data=train_data)
    return preprocessor

def load_train_val_test_data(
    sample=20000,
    val_ratio=0.2,
    test_ratio=0.2,
    test_filter="23.*"
):
    data_path = wod_predictor.__path__[0].replace("wod_predictor", "Data")
    loader = DataLoader(
        root_path=data_path,
        objects= ['open_results','descriptions','benchmark_stats', 'athlete_info']
    )
    data = loader.load()

    val_test_ratio = val_ratio + test_ratio
    splitter = DataSplitter(
        sample=sample, test_ratio=val_test_ratio, test_filter='23.*'
    )
    train_data, val_test_data = splitter.split(data)

    val_test_splitter = DataSplitter(
        test_ratio=test_ratio/val_test_ratio,
        test_filter='23.*'
    )
    val_data, test_data = val_test_splitter.split(val_test_data)
    return train_data, val_data, test_data


