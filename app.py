from flask import Flask, request, render_template
from keras import models  # Import the whole module
from PIL import Image
import numpy as np
import warnings

# Suppress the specific FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load your trained model
model = models.load_model('model_v2.h5')  # Use models to load the model

def predict(image_path):
    img = Image.open(image_path)
    img = img.resize((64, 64))  # Resize to match your model's input shape
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(img_array)
    
    # Suppress the deprecation warning for future versions of NumPy
    predicted_class = int(prediction[0] > 0.5)  # Assuming binary classification

    return predicted_class

@app.route('/')
def home():
    return render_template("upload.html")

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No File Selected"
        
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        
        if file and allowed_file(file.filename):
            file_path = f'static/{file.filename}'
            file.save(file_path)

            print(f"File saved to: {file_path}")

            prediction = predict(file_path)
            print(f"Prediction: {'Real' if prediction == 1 else 'Fake'}")

            return render_template('result.html', 
                                   result='Real' if prediction == 1 else 'Fake', 
                                   image_url=file_path)
        else:
            return "Invalid file type. Please upload an image."

    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)
