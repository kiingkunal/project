from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from efficientnet.tfkeras import preprocess_input, EfficientNetB0

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('/Users/kiingkunal/Desktop/Myproject/best_model.h5')

# Define class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Function to preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to make predictions
def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    predicted_label = class_names[predicted_class]

    # Print additional information for debugging
    print("Predicted Class Index:", predicted_class)
    print("Predicted Class Label:", predicted_label)
    print("Raw Predictions:", predictions)

    return predicted_label

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']

        # Save the file to the server
        file_path = '/Users/kiingkunal/uploads/uploaded_image.jpg'
        file.save(file_path)

        # Make a prediction
        prediction = predict_image(file_path)

        return render_template('index.html', prediction=prediction, image_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
