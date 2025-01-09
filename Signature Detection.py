from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Create the 'uploads' directory if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Load the trained model
model_path = 'D:\Python files\Signature\signature_detection_model.h5'
model = load_model(model_path)

def preprocess_image(image, target_size):
    img_array = cv2.resize(image, target_size)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)  # Make sure to convert the image to RGB format
    return img_array

def draw_bounding_box(image, box_coords):
    image_with_box = image.copy()
    cv2.rectangle(image_with_box, (box_coords[0], box_coords[1]), (box_coords[2], box_coords[3]), (0, 255, 0), 2)
    return image_with_box

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_label = None
    error_message = None
    uploaded_image_path = None

    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            
            image_path = os.path.join('uploads', uploaded_file.filename)
            uploaded_file.save(image_path)
            uploaded_image_path = image_path
            
            image = cv2.imread(image_path)
            image = preprocess_image(image, target_size=(224, 224))
            prediction = model.predict(np.array([image]))
            
            if prediction[0][1] > prediction[0][0] :  # Adjust the threshold as needed
                predicted_label = "Signature"
                # Perform object detection and get bounding box coordinates
                # Replace the following line with your object detection code
                box_coords = (100, 100, 300, 300)  # Example coordinates, adjust as needed
                image_with_box = draw_bounding_box(image, box_coords)
                cv2.imwrite(image_path, image_with_box)
            else:
                predicted_label = "No Signature"
        else:
            error_message = 'No image uploaded'

    return render_template('index.html', predicted_label=predicted_label, error_message=error_message, uploaded_image=uploaded_image_path)


if __name__ == '__main__':
    app.run(debug=True)
