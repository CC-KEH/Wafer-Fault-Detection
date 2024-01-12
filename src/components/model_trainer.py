import pandas as pd 
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from src.exception import CustomException
from src.logger import logging
import os
import sys
from src.utils import evaluate_model,load_object,save_object,upload_file
from src.constant import AWS_S3_BUCKET_NAME

@dataclass
class Model_Trainer_Config:
    model_path = os.path.join('artifacts','model.pkl')

class CustomModel:
    def __init__(self, preprocessing_object, trained_model_object):
        self.preprocessing_object = preprocessing_object

        self.trained_model_object = trained_model_object

    def predict(self, X):
        transformed_feature = self.preprocessing_object.transform(X)

        return self.trained_model_object.predict(transformed_feature)

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"


@dataclass
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = Model_Trainer_Config()
    
    def initiate_model_trainer(self,train_arr,test_arr,preprocessor_path):
        try:
            # Divide dataset into: X_train,X_test,y_train,y_test
            logging.info('Splitting Training, Testing and Target')
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            # Models Dictionary, containing all the candidate models
            models = {
                'Random Forest': RandomForestClassifier(),
                'Gradient Boost': GradientBoostingClassifier(),
                'Adaboost': AdaBoostClassifier(),
                'Decision Tree': DecisionTreeClassifier(),
                'KNN': KNeighborsClassifier(),
                'Logistic Regression': LogisticRegression(),
                'XGBoost': XGBClassifier()
            }
            
            # Training of models, the Model with best accuracy Score on the testing data will be selected for Prediction Pipeline
            logging.info('Extracting Model File path')
            model_report:dict = evaluate_model(X=X_train,y=y_train,models=models)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.vlaues()).index(best_model_score)]
            best_model = models[best_model_name]
            if best_model_score<0.60:
                raise Exception("No best model found")
            
            
            logging.info('Best model found')
            preprocessor_obj = load_object(file_path=preprocessor_path)           
            custom_model = CustomModel(preprocessor_obj,best_model) 
            
            # Save Model
            logging.info(f'Saving model at path: {self.model_trainer_config.model_path}')
            save_object(self.model_trainer_config.model_path,best_model)            
            
            preds = best_model.predict(X_test)
            accuracy = accuracy_score(y_true=y_test,y_pred=preds)
            
            # Upload Model to AWS Bucket
            upload_file(
                from_filename = self.model_trainer_config.model_path,
                to_filename = 'model.pkl',
                bucket_name = AWS_S3_BUCKET_NAME
            )
            
            # Return accuracy Score
            return (best_model_name,accuracy)

        except Exception as e:
            raise CustomException(e,sys)
    
    
        
