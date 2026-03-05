import sys
import os 
import pandas as pd
import numpy as np
from src.utils import save_object
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder, OrdinalEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from sklearn.impute import SimpleImputer
from src.logger import logging
from dataclasses import dataclass


@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation():
    def __init__(self):
        self.datatransformation_config = DataTransformationConfig()
    
    def get_datatranformation_obj(self):
        '''this is responsible for data transformation'''
        
        try:
            numerical_cols = [
                'year', 'unit_price_usd', 'discount_pct', 'units_sold', 'revenue_usd', 'day'
                ]

            OHE_featuers = [
                "region", "category", "sales_channel", "payment_method", "customer_segment", "previous_device_os"
                ]

            TE_featuers = ["country"]

            LE_featuers = [
                "quarter", "customer_age_group", "storage", "month", "day_of_week"
                ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )

            OHE_pipline = Pipeline( steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
                ("scaler",StandardScaler(with_mean=False))
                ])

            target_encoder = TargetEncoder()
            ordinal_encoder = OrdinalEncoder(
                handle_unknown='use_encoded_value',  
                unknown_value=-1  
            )

            logging.info(f"Numerical columns: {numerical_cols}")
            logging.info(f"Target encoding featuers : {TE_featuers}")
            logging.info(f"Label encoding featuers : {LE_featuers}")
            logging.info(f"One Hot encoding featuers : {OHE_featuers}")

            preprocessor = ColumnTransformer([
                ("OneHotEncoder", OHE_pipline, OHE_featuers),
                ("StandardScaler", num_pipeline, numerical_cols),
                ("TargetEncoder", target_encoder, TE_featuers),
                ("OrdinalEncoder", ordinal_encoder, LE_featuers)
        ])
            return preprocessor
        


        except Exception as e:
            raise CustomException(e,sys)
    def intiate_dataTraformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("train and test data loaded ")
            logging.info('obtaining preprocessign object')

            preprocessor_obj  = self.get_datatranformation_obj()


            target_column_name = "return_status"

            le = LabelEncoder()
            train_df[target_column_name] = le.fit_transform(train_df[target_column_name])
            test_df[target_column_name] = le.transform(test_df[target_column_name])

            save_object(
            file_path=os.path.join('artifacts', 'label_encoder.pkl'),
            obj=le
        )

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info('applying preprocessing obj in training and test dataframe')
            target_column_name = "return_status"

            


            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df, target_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)


            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr  = np.c_[input_feature_test_arr,  np.array(target_feature_test_df)]



            save_object(
                file_path=self.datatransformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            logging.info('saved preprocessor object')

            return(
                train_arr,
                test_arr,
                self.datatransformation_config.preprocessor_obj_file_path
            )
        except Exception as e :
            raise CustomException(e,sys)
            


