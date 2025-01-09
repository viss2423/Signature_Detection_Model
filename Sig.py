import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load the saved model
model = tf.keras.models.load_model('D:\\Python files\\Signature\\signature_detection_model.h5')

def preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = preprocess_input(image)
    return image

def predict_and_display(image_path, model, output_path):
    image = preprocess_image(image_path, target_size=(224, 224))
    prediction = model.predict(np.array([image]))

    if prediction[0][1] > prediction[0][0]:  # Signature detected
        # Assuming prediction[0] contains bounding box details (x_min, y_min, x_max, y_max)
        x_min, y_min, x_max, y_max = prediction[0][:4]
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Draw the predicted bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green rectangle
        cv2.putText(image, 'Signature', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save the image with the bounding box
        cv2.imwrite(output_path, image)

    else:
        print("No signature detected.")

image_path_to_predict = 'D:\\Python files\\Signature\\tobacco800_csv_extracted\\aah97e00-page02_1.png'
output_image_path = 'D:\\Python files\\Signature\\predicted_image.png'
predict_and_display(image_path_to_predict, model, output_image_path)
