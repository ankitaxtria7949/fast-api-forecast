from fastapi import FastAPI

app = FastAPI()

@app.get("/check")
def read_root():
    return {"message": "Hello, Vercell!"}
