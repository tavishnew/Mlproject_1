import os
import sys

import pandas as pd
import numpy as np
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV as gridSearchCV
from sklearn.model_selection import RandomizedSearchCV as randomizedSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        pd.to_pickle(obj, file_path)
        logging.info(f"Object saved to {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models, parameters):
    try:    
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            #model.fit(X_train, y_train)
            parameters = parameters.get(model_name, {})

            '''gs = gridSearchCV(model, parameters, cv=3)
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)'''

            rs = randomizedSearchCV(model, parameters, cv=3, n_jobs=-1, verbose=2)
            rs.fit(X_train, y_train)
            model.set_params(**rs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
            logging.info(f"{model_name} - Train Score: {train_model_score}, Test Score: {test_model_score}")

        return report
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)