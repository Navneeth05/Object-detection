from fastapi import FastAPI, UploadFile
import numpy as np
from PIL import Image
from ultralytics import YOLO

app = FastAPI()
model = YOLO("best.pt")

@app.get("/")
def home():
    return {"message": "YOLO Animal Detection API running"}

@app.post("/predict")
async def predict(file: UploadFile):
    image = Image.open(file.file)
    image_np = np.array(image)
    results = model(image_np)
    boxes = results[0].boxes.xyxy.tolist()
    return {"detections": boxes}
