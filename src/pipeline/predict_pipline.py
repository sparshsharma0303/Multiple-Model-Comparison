import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os 

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, model_name ,featuers):
        try:
            model_path = os.path.join('artifacts', f'model_{model_name.replace(' ','_')}.pkl')
            preprocesssor_path = os.path.join('artifacts',"preprocessor.pkl")
            label_encoder_path = os.path.join('artifacts', 'label_encoder.pkl')

            model = load_object(model_path)
            preprocessor = load_object(preprocesssor_path)
            label_encoder = load_object(label_encoder_path)

            

            data_scaled = preprocessor.transform(featuers)
            preds = model.predict(data_scaled)

            return label_encoder.inverse_transform(preds.astype(int))



        except Exception as e :
            raise CustomException(e,sys)

class CustomData:
    def __init__(self, **kwargs):
        self.data = kwargs

    def get_data_as_dataframe(self):
        try:
            return pd.DataFrame([self.data])
        except Exception as e:
            raise CustomException(e, sys)