from PIL import Image
import numpy as np


def predict(file_path):
    try:
        img = Image.open(file_path)
        gray_img = img.convert('L')
        resized_img = gray_img.resize((128,128))
        img_array = np.array(resized_img)
        mean_pixel_img = np.mean(img_array)

        if mean_pixel_img > 100:
            return "AI Generated"
        else:
            return "Real"
    except Exception as e:
        return f"Error in Prediction: {str(e)}"