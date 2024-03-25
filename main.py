from fastapi import FastAPI, File, UploadFile
import uvicorn
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
import json

from starlette.responses import JSONResponse

app = FastAPI()
MODEL = load_model(r'C:\Users\mated\PycharmProjects\ANTIQUE_HUB-API\Models\model_inceptionv3_2 (1).h5')
@app.get("/ping")
async def ping():
 return "Hello, I am alive"



@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # image = read_file_as_image(await file.read())
    # processed_image = preprocess_image(image)
    contents = await file.read()

    img = Image.open(BytesIO(contents))
    img = img.resize((299, 299))  # Resize image to match model's expected sizing
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    output = MODEL.predict(img_array)
    # Define the threshold for classification (e.g., 0.5 for binary classification)
    threshold = 0.5

    # Determine the predicted class label based on the threshold
    predicted_class = "antique" if output[0][0] < threshold else "non_antique"

    response_data= {'class': predicted_class}
    return JSONResponse(content=response_data)




if __name__ == "__main__":
  uvicorn.run(app, host='localhost', port=8000)
