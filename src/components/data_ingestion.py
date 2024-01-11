import pandas as pd 
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import os 
from src.logger import logging
from src.exception import CustomException
from src.utils import export_collection_as_dataframe
import sys
@dataclass
class Data_Ingestion_Config:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_path:str = os.path.join('artifacts','raw.csv')

@dataclass
class DataIngestion:
    def __init__(self):
        self.ingestion_config = Data_Ingestion_Config()
    
    def initiate_data_ingestion(self):
        try:
            # Read Dataset
            df: pd.DataFrame = pd.export_collection_as_dataframe(db_name=MONGO_DATABASE_NAME,collection_name = MONGO_COLLECTION_NAME)
            logging.info('Exported collection as Dataframe')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_path_data_path,index=False,header=True)
            
            # Create a training and testing file            
            train_set,test_set = train_test_split(df,test_size=0.3,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index = False,header = True)
            test_set.to_csv(self.ingestion_config.test_data_path,index = False,header = True)
            logging.info(f'Ingested data from mongodb to {self.ingestion_config.raw_path}')
                
            # Return paths
            return(self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)
                    
        
        except Exception as e:
            raise CustomException(e,sys)