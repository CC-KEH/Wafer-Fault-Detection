import os
import sys
import boto3
import dill
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from src.constant import MONGO_COLLECTION_NAME,MONGO_DATABASE_NAME


def export_collection_as_dataframe(collection_name,db_name):
    try:
        mongo_client = MongoClient(os.getenv('MONGO_DB_URL'))
        collection = mongo_client[db_name][collection_name]
        df = pd.DataFrame(list(collection.find()))
        if '_id' in df.columns.tolist():
            df = df.drop(columns=['_id'],axis=1)
        df.replace({"na",np.nan},inplace=True)
        return df 
       
    except Exception as e:
        raise CustomException(e,sys)

def export_data_into_feature_store_file_path(self)->pd.DataFrame:
    """
    Method Name :   export_data_into_feature_store
    Description :   This method reads data from mongodb and saves it into artifacts. 
    
    Output      :   dataset is returned as a pd.DataFrame
    On Failure  :   Write an exception log and then raise an exception
    
    Version     :   0.1
   
    """
    try:
        logging.info(f"Exporting data from mongodb")
        raw_file_path  = self.data_ingestion_config.artifact_folder
        os.makedirs(raw_file_path,exist_ok=True)
        sensor_data = self.export_collection_as_dataframe(
                                                          collection_name= MONGO_COLLECTION_NAME,
                                                          db_name = MONGO_DATABASE_NAME)
        
        logging.info(f"Saving exported data into feature store file path: {raw_file_path}")
    
        feature_store_file_path = os.path.join(raw_file_path,'wafer_fault.csv')
        sensor_data.to_csv(feature_store_file_path,index=False)
       
        return feature_store_file_path
        
    except Exception as e:
        raise CustomException(e,sys)


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(dir_path,'wb') as obj_file:
            dill.dump(obj,obj_file)
        
    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as obj_file:
            return dill.load(obj_file)
        
    except Exception as e:
        raise CustomException(e,sys)
    
def upload_file(from_filename, to_filename, bucket_name):
    try:
        s3_resource = boto3.resource("s3")

        s3_resource.meta.client.upload_file(from_filename, bucket_name, to_filename)

    except Exception as e:
        raise CustomException(e, sys)


def download_model(bucket_name, bucket_file_name, dest_file_name):
    try:
        s3_client = boto3.client("s3")

        s3_client.download_file(bucket_name, bucket_file_name, dest_file_name)

        return dest_file_name

    except Exception as e:
        raise CustomException(e, sys)





def evaluate_models(X, y, models):
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

