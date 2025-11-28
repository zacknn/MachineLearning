from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# load the trained model
model = load_model('projects/drawing_recognition/model/quickdraw_model.keras')

# categories of drawings
categories = [
    "cat",
    "car",
    "helicopter",
    "star",
    "house",
    "cloud",
    "sun"
]
def predict(image): 
    """
    image: PIL Image (28x28 grayscale)
    """
    img = np.array(image) / 255.0 # normalize
    img = img.reshape((1, 28, 28, 1)) # reshape for model input
    
    prediction = model.predict(img)
    idx = np.argmax(prediction)
    return categories[idx], float(prediction[0][idx])