from networksecurity.exception.exception import NetworkException
from networksecurity.logging.logger import logging
from networksecurity.constant.training_pipeline import MODEL_FILE_NAME, SAVED_MODEL_DIR

import os
import sys


class NetworkModel:
    """
    Base class for all model estimators
    """

    def __init__(self, preprocessor, model):
        """
        Base class for all model estimators
        Args:
            preprocessing_object: preprocessing object
            trained_model: trained model
        """
        try:

            self.preprocessor = preprocessor
            self.model = model

        except Exception as e:
            raise NetworkException(e, sys)

    def predict(self, x):
        """
        Predicts the target values for the given input features.
        Args:
            X: Input features
        Returns:
            Predicted target values
        """
        try:
            logging.info("Entered predict method of NetworkModel class")
            x_transforms = self.preprocessor.transform(x)
            logging.info("Exited predict method of NetworkModel class")
            y_hat = self.model.predict(x_transforms)
            return y_hat
        except Exception as e:
            raise NetworkException(e, sys)
