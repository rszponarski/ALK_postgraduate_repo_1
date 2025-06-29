from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io

class_names = [
    "Black-grass",
    "Charlock",
    "Cleavers",
    "Common Chickweed",
    "Common wheat",
    "Fat Hen",
    "Loose Silky-bent",
    "Maize",
    "Scentless Mayweed",
    "Shepherds Purse",
    "Small-flowered Cranesbill",
    "Sugar beet",
]

app = FastAPI()
model = load_model("plant_seedlings_model.h5")

def read_image(file) -> np.array:
    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims (img_array, axis=0) / 255.0

    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = read_image(io.BytesIO(await file.read()))
    preds = model.predict(img)
    class_idx= np.argmax(preds[0])
    class_name = class_names [class_idx]
    confidence = float(preds[0] [class_idx])
    return {"class": class_name, "confidence": confidence}