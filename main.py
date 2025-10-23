from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.exception.exception import NetworkException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig
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
            
        except Exception as e:
            raise NetworkException(e, sys)
