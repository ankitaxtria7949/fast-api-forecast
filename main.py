from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

@app.post("/demo")
def demo():
    return {"message":"Hellow"}

