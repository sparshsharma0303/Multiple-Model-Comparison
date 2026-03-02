# Basic Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from src.utils import evaluate_models
# Modelling
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelTrainConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainConfig()


    def intiate_model_training(self,train_arr, test_arr):
        try:
            logging.info('entered intiate_model_training block')
            X_train,y_train,X_test,y_test= (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models={
                        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
                        "Ridge Classifier": RidgeClassifier(class_weight='balanced'),
                        "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
                        "Random Forest": RandomForestClassifier(class_weight='balanced'),
                        "Gradient Boosting": GradientBoostingClassifier(),
                        "XGBoost": XGBClassifier(),
                        "CatBoost": CatBoostClassifier(auto_class_weights='Balanced', verbose=0),
                        "Gaussian NB": GaussianNB(),
                        "Bernoulli NB": BernoulliNB(),
                        "KNN": KNeighborsClassifier(),
                        "SVC": SVC(class_weight='balanced', probability=True),
                        "LinearSVC": LinearSVC(class_weight='balanced')
                    }
            model_performance = evaluate_models(X_train, y_train,X_test,y_test,models)

            logging.info(f'model performaces : {model_performance}')

            for model_name, model in models.items():
                model_path = os.path.join('artifacts', f'model_{model_name.replace(" ", "_")}.pkl')
                save_object(file_path=model_path,obj=model)

            logging.info('All models saved successfully')

            summary = []
            for model_name, perf in model_performance.items():
                summary.append({
                    "model": model_name,
                    "train_weighted_f1": round(perf["train"]["weighted_f1"], 4),
                    "test_weighted_f1":  round(perf["test"]["weighted_f1"], 4),
                    "train_macro_f1":    round(perf["train"]["macro_f1"], 4),
                    "test_macro_f1":     round(perf["test"]["macro_f1"], 4),
                })

            return summary



            

        except Exception as e:
            raise CustomException(e,sys)
