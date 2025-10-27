import os
import sys
import certifi
ca = certifi.where()
import pymongo
from dotenv import load_dotenv
load_dotenv()
import pymongo
from networksecurity.exception.exception import NetworkException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import Response, RedirectResponse
from uvicorn import run as app_run
from fastapi.middleware.cors import CORSMiddleware
from networksecurity.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME, DATA_INGESTION_DATABASE_NAME

import pandas as pd
import numpy as np
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)


client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
db = client[DATA_INGESTION_DATABASE_NAME]
collection = db[DATA_INGESTION_COLLECTION_NAME]


app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./template")

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train", tags=["train"])
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")

@app.post("/predict")
async def predict_route(request:Request, file:UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        ##print (df)
        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)
        print(df.iloc[0])
        y_pred = network_model.predict(df)
        print(y_pred)
        df['predicted_column'] = y_pred
        print(df['predicted_column'])
        #df['predicted_column].replace(-1, 0)
        #return df.to_json()
        df.to_csv("prediction_output/output.csv")
        table_html = df.to_html(classes='table table-striped')
        #print(table_html))
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
    
    except Exception as e:
        raise NetworkException(e, sys)


if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8080)