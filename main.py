from io import BytesIO
import torch
from breed_classification.models import EnsembleModel
from breed_classification.utils import predict
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import uvicorn
import os


app = FastAPI()

# Get model path name
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))
model_name_path = os.path.join(
    parent_path, "breed_classification", "weights", "ensemble_model.pt"
)

loaded_ensemble_model = EnsembleModel()
loaded_ensemble_model.load_state_dict(torch.load(model_name_path, map_location="cpu"))


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image


@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    results = dict()
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    # image = read_imagefile(await file.read())
    probs, classes, cls_name = predict(loaded_ensemble_model, await file.read())
    for prob, class_name in zip(probs[0], classes[0]):
        results[cls_name[class_name]] = round(prob, 3)
    return results


if __name__ == "__main__":
    # uvicorn.run(app, debug=True)
    uvicorn.run("main:app", host="0.0.0.0", port=80, reload=False, debug=False)
