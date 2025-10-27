import os
import sys
from networksecurity.exception.exception import NetworkException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import (
    TrainingPipelineConfig,
    ModelTrainerConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
)
from networksecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer


class TrainingPipeline:
    def __init__(self):
        try:
            self.training_pipeline_config = TrainingPipelineConfig()
        except Exception as e:
            raise NetworkException(e, sys)

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("📥 Starting data ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"✅ Data ingestion completed: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkException(e, sys)

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("🧩 Starting data validation")
            data_validation = DataValidation(
                data_validation_config=data_validation_config, 
                data_ingestion_artifact=data_ingestion_artifact
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(f"✅ Data validation completed: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise NetworkException(e, sys)

    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("⚙️ Starting data transformation")
            data_transformation = DataTransformation(
                data_validation_artifact=data_validation_artifact, 
                data_transformation_config=data_transformation_config
            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info(f"✅ Data transformation completed: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise NetworkException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("🤖 Starting model training")
            model_trainer = ModelTrainer(
                model_trainer_config=model_trainer_config, 
                data_transformation_artifact=data_transformation_artifact
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info(f"✅ Model training completed: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise NetworkException(e, sys)

    def run_pipeline(self):
        try:
            logging.info("🚀 Training pipeline started")

            # === 1️⃣ Run pipeline stages ===
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)

            # === 2️⃣ Sync artifacts + models to S3 ===
            cloud_sync = CloudSyncManager(self.training_pipeline_config)
            cloud_sync.sync_artifact_dir_to_s3()
            cloud_sync.sync_saved_model_dir_to_s3()

            logging.info("🎯 Training pipeline completed and synced to S3 successfully!")
            return model_trainer_artifact
        except Exception as e:
            raise NetworkException(e, sys)


class CloudSyncManager:
    def __init__(self, training_pipeline_config):
        self.training_pipeline_config = training_pipeline_config
        self.timestamp = getattr(training_pipeline_config, "timestamp", "latest")

    def sync_artifact_dir_to_s3(self):
        """Sync local artifact directory to S3"""
        try:
            if self.training_pipeline_config.sync_artifact_dir:
                s3_uri = f"s3://{self.training_pipeline_config.s3_bucket_name}/artifact/{self.timestamp}"
                cmd = f"aws s3 sync {self.training_pipeline_config.artifact_dir} {s3_uri} --profile {self.training_pipeline_config.aws_profile_name}"
                logging.info(f"☁️ Syncing artifact directory to {s3_uri} ...")
                os.system(cmd)
                logging.info("✅ Artifact directory synced to S3 successfully")
            else:
                logging.info("Artifact directory syncing is disabled")
        except Exception as e:
            raise NetworkException(e, sys)

    def sync_saved_model_dir_to_s3(self):
        """Sync local saved model directory to S3"""
        try:
            if self.training_pipeline_config.sync_saved_model_dir:
                s3_uri = f"s3://{self.training_pipeline_config.s3_bucket_name}/final_model/{self.timestamp}"
                cmd = f"aws s3 sync {self.training_pipeline_config.saved_model_dir} {s3_uri} --profile {self.training_pipeline_config.aws_profile_name}"
                logging.info(f"☁️ Syncing saved model directory to {s3_uri} ...")
                os.system(cmd)
                logging.info("✅ Saved model directory synced to S3 successfully")
            else:
                logging.info("Saved model directory syncing is disabled")
        except Exception as e:
            raise NetworkException(e, sys)


if __name__ == "__main__":
    training_pipeline = TrainingPipeline()
    training_pipeline.run_pipeline()
