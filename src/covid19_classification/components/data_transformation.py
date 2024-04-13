import os 
import sys 
import pandas as pd
import numpy as np 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.covid19_classification.exception import CustomException
from src.covid19_classification.logger import logging
from src.covid19_classification.utils import save_object

from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTranformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):


        try:
            cols = ['Cough_symptoms', 'Fever', 'Sore_throat', 'Shortness_of_breath','Headache', 'Age_60_above', 'Sex', 'Known_contact']

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder())
                ]
            )

            preprocessor = ColumnTransformer([
                ("cat_pipeline", cat_pipeline, cols)]
            ) 
            
            logging.info(f"Returned Transformed Columns:{cols}")

            return preprocessor
        
            

        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading the train and test dataframe")

            preprocessing_obj = self.get_data_transformer_object()

            target_col_name="Corona"
            #mapping the target feature values for train and test df
            train_df[target_col_name] = train_df[target_col_name].map({'negative' : 0, 'positive' : 1})
            test_df[target_col_name] = train_df[target_col_name].map({'negative' : 0, 'positive' : 1})

            #dividing the train dataset in independent and target features
            input_features_train_df=train_df.drop(columns=[target_col_name], axis=1)
            target_features_train_df=train_df[target_col_name]

            #dividing the test dataset in independent and target features
            input_features_test_df=test_df.drop(columns=[target_col_name], axis=1)
            target_features_test_df=test_df[target_col_name]

            logging.info("Applying Preprocessing on training and test dataframe")

            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessing_obj.transform(input_features_test_df)

            train_arr =np.c_[input_features_train_arr, np.array(target_features_train_df)]

            test_arr = np.c_[input_features_test_arr, np.array(target_features_test_df)]

            logging.info("Transformation done")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Saved preprocessing object")

            return(
                train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)




