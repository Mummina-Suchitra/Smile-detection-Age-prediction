from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your models
smile_model = load_model("models/smile_model.h5")
age_model = load_model("models/age_model.h5")

def predict_image(img_path):
    """
    Input: Path to image
    Output: smile_result (Smiling/Not Smiling), age_result (Age: XX years)
    """
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(100, 100))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Smile prediction
    smile_pred = smile_model.predict(img_array)
    smile_class = "Smiling 😊" if np.argmax(smile_pred) == 1 else "Not Smiling 😐"

    # Age prediction (exact age)
    age_pred = age_model.predict(img_array)
    age_years = int(age_pred[0][0])
    age_class = f"Age: {age_years} years"

    return smile_class, age_class
