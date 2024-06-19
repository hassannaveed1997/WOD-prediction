def generate_meta_data(data):
    meta_data = {}

    meta_data['idx_to_workout_name'] = get_workout_name_mapping(data)

    return meta_data


def get_workout_name_mapping(data):
    workout_cols = data.columns[data.columns.str.contains("workout")]

    if len(workout_cols) == 1:
        workout_name_mapping = data[workout_cols[0]]
    else:
        # get argmax
        workout_name_mapping = data[workout_cols].idxmax(axis=1)
    return workout_name_mapping
