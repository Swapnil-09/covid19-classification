import os
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.covid19_classification.logger import logging
from src.covid19_classification.exception import CustomException
from src.covid19_classification.utils import save_object, evaluate_models

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):

        try:
            logging.info("Split training and test data inputs")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Logistic Regression" : LogisticRegression(),
                "Decision Tree" : DecisionTreeClassifier(),
                "Random Forest" : RandomForestClassifier(),
                "XGB" : XGBClassifier(),
                "SVC" : SVC(),
                "Adaboost" : AdaBoostClassifier()
            }

            params={
                "Logistic Regression" : {

                },
                "Decision Tree" : {
                    'max_features':['sqrt','log2']
                },
                "Random Forest" : {
                    'n_estimators': [8,16,32,64,128,256],
                    'max_features' : [1, 3, 5, 7]
                },
                "XGB" : {
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "SVC" : {

                },
                "Adaboost" : {
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }
            
            model_report:dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best found model on both training and test data : {best_model}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
        


        except Exception as e:
            raise CustomException(e, sys)