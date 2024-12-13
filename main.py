from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse,RedirectResponse
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Plant Disease Analysis API")

# Load model and labels at startup
MODEL_PATH = "keras_model1.h5"
LABELS_PATH = "labels.txt"

model = load_model(MODEL_PATH, compile=False)
class_names = open(LABELS_PATH, "r").readlines()


@app.get("/",include_in_schema=False)
def index():
    return RedirectResponse("/docs", status_code=308)

@app.post("/predict/")
def predict_image(file: UploadFile = File(...)):
    try:
        # Check if the uploaded file has a valid content type
        # if not file.filename.endswith((".jpg", ".jpeg", ".png")):
        #     raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

        # Open and preprocess the image
        print(file.content_type)
        image = Image.open(file.file).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Perform prediction
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()  # Strip newline characters
        confidence_score = float(prediction[0][index])

        # Return the prediction as JSON
        return JSONResponse(
            content={
                "class_name": class_name,
                "confidence_score": confidence_score
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

