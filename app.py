from fastapi import FastAPI, UploadFile, File
import shutil, os
from detector import detect_deepfake

app = FastAPI()

UPLOAD_FOLDER="uploads"
os.makedirs(UPLOAD_FOLDER,exist_ok=True)

@app.get("/")
def home():
    return {"RealityShield":"Running"}

@app.post("/scan")
async def scan(file: UploadFile = File(...)):
    path=f"{UPLOAD_FOLDER}/{file.filename}"

    with open(path,"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)

    result,score = detect_deepfake(path)

    return {
        "prediction":result,
        "confidence_score":score
    }