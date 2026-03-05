import pickle
import sys

import os, pandas as pd, numpy as np
from src.logger import logging
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.exception import CustomException
from sklearn.model_selection import RandomizedSearchCV


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj :
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(X_train, y_train,X_test,y_test,models,param_grids=None):
    try:
        model_performances = {}

        for model_name,model in models.items():

            if param_grids and model_name in param_grids:
                rs = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grids[model_name],
                    n_iter=10,
                    cv=2,
                    scoring='f1_macro',   # optimize for macro F1
                    random_state=42,
                    n_jobs=-1,
                    verbose=0
                )
                rs.fit(X_train, y_train)
                model = rs.best_estimator_
                logging.info(f'{model_name} best params: {rs.best_params_}')
            else:
                model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            # "weighted_f1": model_train_f1score_weighted,
            # "macro_f1": model_train_f1score_macro,
            # "confusion_matrix": model_train_confusion_matrix,
            # "classification_report": model_train_classification_report

            model_train_f1score_weighted = f1_score(y_train, y_train_pred , average='weighted',zero_division=0)
            model_train_f1score_macro = f1_score(y_train, y_train_pred , average='macro',zero_division= 0 )
            model_train_confusion_matrix = confusion_matrix(y_train, y_train_pred)
            model_train_classification_report = classification_report(y_train, y_train_pred)

            model_test_f1score_weighted = f1_score(y_test, y_test_pred , average='weighted',zero_division=0)
            model_test_f1score_macro = f1_score(y_test, y_test_pred , average='macro',zero_division= 0 )
            model_test_confusion_matrix = confusion_matrix(y_test, y_test_pred)
            model_test_classification_report = classification_report(y_test, y_test_pred)

            model_performances[model_name] = {
            "train" : {
            "weighted_f1": model_train_f1score_weighted,
            "macro_f1": model_train_f1score_macro,
            "confusion_matrix": model_train_confusion_matrix,
            "classification_report": model_train_classification_report
            },
            "test" : {
            "weighted_f1": model_test_f1score_weighted,
            "macro_f1": model_test_f1score_macro,
            "confusion_matrix": model_test_confusion_matrix,
            "classification_report": model_test_classification_report
            }
        }
        return model_performances
    except Exception as e :
        raise CustomException(e,sys)

        

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
