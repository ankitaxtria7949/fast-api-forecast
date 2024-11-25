from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
import json
from LinearRegression import LinearRegression
from datetime import datetime
from fastapi import File, UploadFile, Form
from typing import Optional
import csv
from io import StringIO
from LogLinearRegression import log_linear_regression
from Naive import naive_forecast
from SeasonalNaive import seasonal_naive_forecast
from Average import average_forecast
from ETS import ETS

app=FastAPI()
# Allow React app on localhost:3000 to communicate with the FastAPI backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Can replace with specific frontend URL, like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods like GET, POST, etc.
    allow_headers=["*"],  # Allows all headers
)
 

 
 

    
@app.get("/")
def read_root():
    return {"message": "Hello World"}


@app.post("/upload")
async def forecast_sales(file:UploadFile=File(...), selectedSheet: Optional[str] = Form(None), historyFromDate: Optional[str] = Form(None), historyToDate: Optional[str] = Form(None), selectedFromDate: Optional[str] = Form(None), selectedToDate: Optional[str] = Form(None)): 
    contents = await file.read()
    print(historyFromDate, historyToDate, selectedFromDate, selectedToDate)
    decoded = contents.decode('utf-8')
    csv_reader = csv.reader(StringIO(decoded), delimiter=',')
    data = [row for row in csv_reader]
    if selectedSheet == 'Linear Regression':
        forecast_val, dt = LinearRegression(data, historyFromDate, historyToDate, selectedFromDate, selectedToDate)
    elif selectedSheet == 'Log Linear Regression':
        forecast_val, dt = log_linear_regression(data, historyFromDate, historyToDate, selectedFromDate, selectedToDate)
    elif selectedSheet == 'Seasonal Naive':
        forecast_val, dt = seasonal_naive_forecast(data, historyFromDate, historyToDate, selectedFromDate, selectedToDate)  
    elif selectedSheet == "Naive":
        forecast_val, dt = naive_forecast(data, historyFromDate, historyToDate, selectedFromDate, selectedToDate)
    elif selectedSheet == "Average":
        forecast_val, dt = average_forecast(data, historyFromDate, historyToDate, selectedFromDate, selectedToDate)
    elif selectedSheet == "Additive Trend-Additive Seasonality":
        forecast_val, dt = ETS(data, historyFromDate, historyToDate, selectedFromDate, selectedToDate)
    return {"forecast": forecast_val, "dt": dt,"filename": file.filename, "historyFromDate" : historyFromDate,"historyToDate" : historyToDate,"selectedFromDate" : selectedFromDate,"selectedToDate" : selectedToDate}

