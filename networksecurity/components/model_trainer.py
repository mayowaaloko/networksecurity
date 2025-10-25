import sys
import os
from networksecurity.exception.exception import NetworkException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)

from networksecurity.utils.main_utils.utils import (
    load_numpy_array_data,
    load_object,
    save_object,
    evaluate_models,
)
from networksecurity.utils.ml_utils.metric.classification_metrics import (
    get_classification_score,
)
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)

import mlflow
import dagshub

# ✅ Initialize MLflow tracking with DagsHub
dagshub.init(repo_owner="mayowaaloko", repo_name="networksecurity", mlflow=True)
mlflow.set_experiment("Network_Security_Models")


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkException(e, sys)

    def track_mlflow(
        self, model_name, model, train_metric, test_metric, local_model_path
    ):
        """
        Logs model parameters, metrics, and local artifact to DagsHub MLflow.
        Avoids model upload due to DagsHub’s unsupported endpoint.
        """
        try:
            with mlflow.start_run(run_name=model_name):
                mlflow.log_param("model_name", model_name)

                # Log model hyperparameters
                for param_name, param_value in model.get_params().items():
                    mlflow.log_param(param_name, param_value)

                # Log training metrics
                mlflow.log_metric("train_precision", train_metric.precision_score)
                mlflow.log_metric("train_recall", train_metric.recall_score)
                mlflow.log_metric("train_f1_score", train_metric.f1_score)

                # Log testing metrics
                mlflow.log_metric("test_precision", test_metric.precision_score)
                mlflow.log_metric("test_recall", test_metric.recall_score)
                mlflow.log_metric("test_f1_score", test_metric.f1_score)

                # ✅ Log model file as a plain artifact (NOT via mlflow.sklearn.log_model)
                mlflow.log_artifact(local_model_path)

                logging.info(
                    f"✅ Metrics and model artifact logged to DagsHub for {model_name}."
                )

        except Exception as e:
            raise NetworkException(e, sys)

    def train_model(self, x_train, y_train, x_test, y_test):
        try:
            models = {
                "LogisticRegression": LogisticRegression(verbose=1),
                "DecisionTree": DecisionTreeClassifier(),
                "RandomForest": RandomForestClassifier(verbose=1),
                "GradientBoosting": GradientBoostingClassifier(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
                "KNeighbors": KNeighborsClassifier(),
            }

            # ✅ Original parameter grids restored
            params = {
                "LogisticRegression": {
                    "C": [0.01, 0.1, 1, 10, 100],
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear", "saga"],
                },
                "DecisionTree": {
                    "criterion": ["gini", "entropy"],
                },
                "RandomForest": {
                    "criterion": ["gini", "entropy", "log_loss"],
                },
                "GradientBoosting": {
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9],
                },
                "AdaBoost": {
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                },
                "KNeighbors": {
                    "n_neighbors": [8, 16, 32, 64, 128, 256],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                },
            }

            # Evaluate models
            model_report: dict = evaluate_models(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                params=params,
            )

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # Predictions
            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)

            # Compute metrics
            classification_train_metric = get_classification_score(
                y_true=y_train, y_pred=y_train_pred
            )
            classification_test_metric = get_classification_score(
                y_true=y_test, y_pred=y_test_pred
            )

            diff = abs(
                classification_train_metric.f1_score
                - classification_test_metric.f1_score
            )
            if diff > self.model_trainer_config.overfitting_underfitting_threshold:
                raise Exception(
                    f"Train-test F1 diff {diff} exceeds threshold {self.model_trainer_config.overfitting_underfitting_threshold}"
                )

            # Save model locally
            preprocessor = load_object(
                file_path=self.data_transformation_artifact.preprocessed_object_file_path
            )
            model_dir_path = os.path.dirname(
                self.model_trainer_config.trained_model_file_path
            )
            os.makedirs(model_dir_path, exist_ok=True)

            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
            save_object(
                self.model_trainer_config.trained_model_file_path, obj=network_model
            )

            # Backup pure model too
            os.makedirs("final_model", exist_ok=True)
            save_object("final_model/model.pkl", best_model)

            # ✅ Log only metrics and artifact to DagsHub (safe)
            self.track_mlflow(
                best_model_name,
                best_model,
                classification_train_metric,
                classification_test_metric,
                self.model_trainer_config.trained_model_file_path,
            )

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric,
            )

            logging.info(f"✅ Model trainer artifact created: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise NetworkException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("🚀 Starting model trainer...")
            train_file_path = (
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_file_path = (
                self.data_transformation_artifact.transformed_test_file_path
            )

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)
            return model_trainer_artifact

        except Exception as e:
            raise NetworkException(e, sys)
