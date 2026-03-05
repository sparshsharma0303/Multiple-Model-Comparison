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
from src.utils import save_object,evaluate_models
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

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
            # smt = SMOTETomek(
            # smote=SMOTE(sampling_strategy=0.3, random_state=42),  # don't fully balance
            # random_state=42)

            # X_train, y_train = smt.fit_resample(X_train, y_train)
            # logging.info(f'After SMOTETomek - class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}')
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
            param_grids ={
                "Gradient Boosting": {
                    "n_estimators":      [100, 200, 300],
                    "max_depth":         [3, 4, 5],
                    "learning_rate":     [0.05, 0.1, 0.2],
                    "subsample":         [0.7, 0.8, 1.0],
                    "min_samples_split": [2, 5, 10]
                },
                "Decision Tree": {
                    "max_depth":         [3, 5, 7, 10, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf":  [1, 2, 4],
                    "criterion":         ["gini", "entropy"]
                }
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
        
if __name__ == "__main__":
    ingestion_obj = DataIngestion()

    train_data,test_data = ingestion_obj.initiate_data_ingestion()

    transformation_obj = DataTransformation()
    train_arr, test_arr, preprocessor_obj_path = transformation_obj.intiate_dataTraformation(train_data,test_data)

    trainer_obj = ModelTrainer()

    summary = trainer_obj.intiate_model_training(train_arr,test_arr)

    print(summary)