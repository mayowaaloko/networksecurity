from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.exception.exception import NetworkException
from networksecurity.logging.logger import logging
from networksecurity.components.data_validation import DataValidation
from networksecurity.entity.config_entity import DataIngestionConfig, DataValidationConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig
import sys




    
if __name__ == "__main__":
        try:
            trainingpipeline_config = TrainingPipelineConfig()
            dataingestion_config = DataIngestionConfig(trainingpipeline_config)
            data_ingestion = DataIngestion(dataingestion_config)
            logging.info("initiate the data ingestion")
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("completed the data ingestion")
            print(data_ingestion_artifact)
            
            data_validation_config = DataValidationConfig(trainingpipeline_config)
            data_validation = DataValidation(data_validation_config, data_ingestion_artifact)
            logging.info("initiate the data validation")
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("completed the data validation")
            print(data_validation_artifact)

            
        except Exception as e:
            raise NetworkException(e, sys)
