from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI


app=FastAPI()


 

    
@app.get("/")
def read_root():
    return {"message": "Hello World"}