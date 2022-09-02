import json
import pickle
import numpy as np

global __data_columns
global __locations
global __model
with open("./artifacts/columns.json", "r") as f:
    __data_columns = json.load(f)["data_columns"]
    __locations = __data_columns[3:]

with open("./artifacts/real_estate_price_prediction.pickle", "rb") as f:
    __model = pickle.load(f)


def predict_price(location, size, sqft, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1
    x = np.zeros(len(__data_columns))
    x[0] = size
    x[1] = sqft
    x[2] = bath
    if loc_index >= 0:
        x[loc_index] = 1
    return round(__model.predict([x])[0], 2)


def load_saved_artifacts():
    print("Loading saved artifacts....start")
    global __data_columns
    global __locations
    global __model
    with open("./artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)["data_columns"]
        __locations = __data_columns[3:]

    with open("./artifacts/real_estate_price_prediction.pickle", "rb") as f:
        __model = pickle.load(f)
    print("Loading saved artifacts....done")


def get_location_names():
    return __locations


if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_names())
    print(predict_price('Indira Nagar', 2, 1000, 2))
    print(predict_price('1st Phase JP Nagar', 2, 1000, 2))
    print(predict_price('Indira Nagar', 3, 1000, 3))