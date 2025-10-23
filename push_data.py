import os
import sys
import json
import certifi
import pandas as pd
import pymongo
from dotenv import load_dotenv
from networksecurity.exception.exception import NetworkException
from networksecurity.logging.logger import logging

load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
ca = certifi.where()


class NetworkDataHandler:
    def __init__(self, database_name: str):
        try:
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
            self.db = self.mongo_client[database_name]
        except Exception as e:
            raise NetworkException(e, sys)

    def csv_to_records(self, file_path: str):
        try:
            df = pd.read_csv(file_path)
            df.reset_index(drop=True, inplace=True)
            return list(json.loads(df.T.to_json()).values())
        except Exception as e:
            raise NetworkException(e, sys)

    def insert_records(self, collection_name: str, records: list, replace=False):
        try:
            collection = self.db[collection_name]
            if replace:
                collection.delete_many({})
            collection.insert_many(records)
            return len(records)
        except Exception as e:
            raise NetworkException(e, sys)


if __name__ == "__main__":
    FILE_PATH = "Network_Data/phisingData.csv"
    DATABASE = "MAYOWA"
    COLLECTION = "NetworkData"

    handler = NetworkDataHandler(DATABASE)
    records = handler.csv_to_records(FILE_PATH)
    print(records)
    count = handler.insert_records(COLLECTION, records, replace=True)
    print(f"{count} records inserted successfully.")
    logging.info(f"{count} records inserted successfully.")
    print("Document count in MongoDB:", handler.db[COLLECTION].count_documents({}))
