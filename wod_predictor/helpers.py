import os
import wod_predictor

def get_base_path(levels_up = 1):
    wod_prediction_path = os.path.dirname(wod_predictor.__file__)
    for _level in range(1,levels_up):
        wod_prediction_path = os.path.dirname(wod_prediction_path)
    return wod_prediction_path
