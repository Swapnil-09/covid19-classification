from src.covid19_classification.logger import logging
from src.covid19_classification.exception import CustomException
from src.covid19_classification.components.data_ingestion import DataIngestion
from src.covid19_classification.components.data_transformation import DataTranformation
from src.covid19_classification.components.model_trainer import ModelTrainer
import sys

if __name__=="__main__":
    logging.info("The execution has started")


    try:
        data_ingestion=DataIngestion()
        train_data_path, test_data_path=data_ingestion.initiate_data_ingestion()

        data_transformation=DataTranformation()
        train_arr, test_arr , _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr, test_arr))


    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)
