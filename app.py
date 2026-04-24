from pathlib import Path
import io
import numpy as np
import tensorflow as tf
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from model import CurrencyCNN

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
H5_MODEL_PATH = BASE_DIR / "model.h5"
PTH_MODEL_PATH = BASE_DIR / "model.pth"

model = None
use_tensorflow = False
class_names = []

if H5_MODEL_PATH.exists():
    model = tf.keras.models.load_model(str(H5_MODEL_PATH))
    use_tensorflow = True
    if hasattr(model, "class_names"):
        class_names = list(model.class_names)
    else:
        output_dim = model.output_shape[-1]
        class_names = [f"class_{i}" for i in range(output_dim)]
elif PTH_MODEL_PATH.exists():
    checkpoint = torch.load(PTH_MODEL_PATH, map_location=torch.device("cpu"))
    class_names = checkpoint.get("classes", [])
    # Use simple CNN instead of ResNet18
    num_classes = len(class_names) if class_names else 2
    model = CurrencyCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
else:
    available = [p.name for p in BASE_DIR.glob("model.*")]
    raise FileNotFoundError(
        f"No trained model found in {BASE_DIR}.\n"
        f"Expected one of: model.h5 or model.pth.\n"
        f"Available model files: {available or 'none'}.\n"
        f"Run train.py to create model.pth, or place a TensorFlow model.h5 file in the project root."
    )

if not class_names and not use_tensorflow:
    output_dim = model.fc2.out_features
    class_names = [f"class_{i}" for i in range(output_dim)]

# If using PyTorch, we keep the TensorFlow preprocess output as NHWC, then convert.

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    array = np.array(image, dtype=np.float32) / 255.0
    array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return np.expand_dims(array, axis=0)

@app.get("/")
def root():
    return {"message": "Upload an image to /predict"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor = preprocess_image(image)

    if use_tensorflow:
        predictions = model.predict(input_tensor)
        prob = tf.nn.softmax(predictions[0]).numpy()
    else:
        tensor = torch.from_numpy(np.transpose(input_tensor, (0, 3, 1, 2)))
        with torch.no_grad():
            outputs = model(tensor)
            prob = torch.softmax(outputs, dim=1)[0].numpy()

    pred_index = int(np.argmax(prob))

    return {
        "label": class_names[pred_index],
        "confidence": float(prob[pred_index]),
        "classes": class_names,
    }
