from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import File, UploadFile, Form
from typing import Optional
import csv
from io import StringIO
from LinearRegression import LinearRegression
from LogLinear import LogLinear
from Average import Average
from Holt import Holt

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
    decoded = contents.decode('utf-8')
    csv_reader = csv.reader(StringIO(decoded), delimiter=',')
    data = [row for row in csv_reader]
    if len(data[0]) == 2 and len(data) > 2:
        dates = [entry[0] for entry in data]
        values = [entry[1] for entry in data]
        data = [dates, values]
    print(data)
    if selectedSheet == 'Linear Regression':
        forecast_val, dt, metrics = LinearRegression(data, historyFromDate, historyToDate, selectedFromDate, selectedToDate)
    elif selectedSheet == 'Log Linear Regression':
        forecast_val, dt, metrics = LogLinear(data, historyFromDate, historyToDate, selectedFromDate, selectedToDate)   
    elif selectedSheet == 'Average':
        forecast_val, dt, metrics = Average(data, historyFromDate, historyToDate, selectedFromDate, selectedToDate)
    elif selectedSheet == 'Holt':
        forecast_val, dt, metrics = Holt(data, historyFromDate, historyToDate, selectedFromDate, selectedToDate)
    return {"forecast": forecast_val, "dt": dt, "metrices": metrics, "filename": file.filename, "historyFromDate" : historyFromDate,"historyToDate" : historyToDate,"selectedFromDate" : selectedFromDate,"selectedToDate" : selectedToDate}


