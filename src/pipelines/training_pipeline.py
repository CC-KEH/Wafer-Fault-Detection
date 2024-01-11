import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def __init__(self) -> None:
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
        
    def start_pipeline(self):
        try:
            # Data Ingestion
            train_path,test_path = self.data_ingestion.initiate_data_ingestion()
            
            # Data Transformation
            train_arr,test_arr,preprocessor_path = self.data_transformation.initiate_transformation()
            
            # Model Train
            model_name,model_r2_score = self.model_trainer.initiate_model_trainer(train_arr,test_arr,preprocessor_path)
            
            print(f'Training Completed. Train Model Name: {model_name}, Score: {model_r2_score}')
            
            
        except Exception as e:
            pass
            # raise CustomException(e,sys)
        