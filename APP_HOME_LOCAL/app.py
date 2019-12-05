import pathlib
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

from flask import Flask, request # for building API
import json
from datetime import datetime

from ml_pipeline import pipeline
from ml_pipeline import preprocessors as pp


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent # root package dir
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_model' # trained model model dir
DATASET_DIR = PACKAGE_ROOT / 'datasets' # datasets dir

TESTING_DATA_FILE = DATASET_DIR / 'test.csv' # test data dir
TRAINING_DATA_FILE = DATASET_DIR / 'train.csv' # train data dir
TARGET = 'SalePrice'

#pipeline_file_name = TRAINED_MODEL_DIR / 'regression_model.pkl' # ml pipeline [processing + prediction]
#_price_pipe = joblib.load(pipeline_file_name)



FEATURES = ['MSSubClass', 'MSZoning', 'Neighborhood', 'OverallQual',
            'OverallCond', 'YearRemodAdd', 'RoofStyle', 'MasVnrType',
            'BsmtQual', 'BsmtExposure', 'HeatingQC', 'CentralAir',
            '1stFlrSF', 'GrLivArea', 'BsmtFullBath', 'KitchenQual',
            'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageFinish',
            'GarageCars', 'PavedDrive', 'LotFrontage',
            # this variable is only to calculate temporal variable:
            'YrSold']

FEATURES_TYPES = {
    'MSSubClass': 'int64',
    'MSZoning': 'object',
    'Neighborhood': 'object',
    'OverallQual': 'int64',
    'OverallCond': 'int64',
    'YearRemodAdd': 'int64',
    'RoofStyle': 'object',
    'MasVnrType': 'object',
    'BsmtQual': 'object',
    'BsmtExposure': 'object',
    'HeatingQC': 'object',
    'CentralAir': 'object',
    '1stFlrSF': 'int64',
    'GrLivArea': 'int64',
    'BsmtFullBath': 'int64',
    'KitchenQual': 'object',
    'Fireplaces': 'int64',
    'FireplaceQu': 'object',
    'GarageType': 'object',
    'GarageFinish': 'object',
    'GarageCars': 'int64',
    'PavedDrive': 'object',
    'LotFrontage': 'int64',
    'YrSold': 'int64'
}

app = Flask(__name__) # create an class instance of Flask

def save_pipeline(*, pipeline_to_persist) -> None:
    """Persist the pipeline."""

    save_file_name = 'regression_model.pkl'
    save_path = TRAINED_MODEL_DIR / save_file_name
    joblib.dump(pipeline_to_persist, save_path)

    print('saved pipeline')

@app.route("/retrain_model", methods=["GET"])
def retrain_model():
    """Train the model."""

    if request.method == "GET":

        # create reponse data template
        now = datetime.utcnow()
        response_data = {
            "timeStamp": now.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3],
            "response": {}
            }
    
        # read training data
        data = pd.read_csv(TRAINING_DATA_FILE)

        # divide train and test
        X_train, X_test, y_train, y_test = train_test_split(
            data[FEATURES],
            data[TARGET],
            test_size=0.1,
            random_state=0)  # we are setting the seed here

        # transform the target
        y_train = np.log(y_train)
        y_test = np.log(y_test)

        pipeline.price_pipe.fit(X_train[FEATURES],
                                y_train)

        save_pipeline(pipeline_to_persist=pipeline.price_pipe)

        response_data["response"] = {"status":"pipeline is trained and ready to use for prediction", "regression_model_version": "1.0.0"}
        return json.dumps(response_data), 200


@app.route("/make_prediction", methods=["POST"])
def make_prediction():
    """Make a prediction using the saved model pipeline."""

    if request.method == 'POST':

        #get arguments from the request
        data_json = request.get_json()

        data = pd.DataFrame.from_dict(data_json, orient='index').transpose() \
            [FEATURES].astype(FEATURES_TYPES) 

        # create reponse data template
        now = datetime.utcnow()
        response_data = {
            "timeStamp": now.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3],
            "response": {}
            }

        pipeline_file_name = TRAINED_MODEL_DIR / 'regression_model.pkl' # ml pipeline [processing + prediction]
        _price_pipe = joblib.load(pipeline_file_name)

        prediction = _price_pipe.predict(data[FEATURES])
        output = np.exp(prediction).tolist()
        response_data["response"] = {'predictions': output}

        return json.dumps(response_data), 200

if __name__ == "__main__":
    app.run()


