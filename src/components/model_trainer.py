import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainingConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainingConfig()


    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("split train and test input data")


            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1] 
            )

            models={
                "Random Forest":RandomForestRegressor(),
                "decision tree": DecisionTreeRegressor(),
                "adaboost": AdaBoostRegressor(),
                "catboost":CatBoostRegressor(),
                "Linear regression": LinearRegression(),
                "Knearest neighbour": KNeighborsRegressor(),
                "XGBoost regressor":XGBRegressor(),
                "gradient boost":GradientBoostingRegressor(),
            }

            model_report: dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                              models=models)


            #to get best model score from dict
            best_model_score=max(sorted(model_report.values()))
           
            #to get best model name from dict
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found",sys)  

            logging.info("best found model on both training and test dataset")         


            save_object(

                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
                
            )

            y_predicted=best_model.predict(X_test)

            r2_squared=r2_score(y_test,y_predicted)
            return r2_squared

        ## yahan tk phle done \



        except Exception as e:
            raise CustomException(e,sys)
             
