import numpy as np
import os
import joblib  # For loading the RandomForest model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import load_model

# Path to the folder containing the images
image_folder_path = '/Users/nachuthenappan/aihack23/Training-Set-1'

# Load the trained Random Forest model (saved as .pkl)
clf = joblib.load('/Users/nachuthenappan/aihack23/7random_forest_model.pkl')  # Use the correct path to your saved RandomForest model

# Function to load and preprocess the image
def process_image(image_path):
    """Load and preprocess an image for prediction."""
    img = image.load_img(image_path, target_size=(224, 224))  # Resize image to (224, 224)
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess image for VGG19
    return img_array


# Function to extract features using VGG19 (as the updated feature extractor)
def extract_features(image_path):
    """Extract features from an image using VGG19."""
    img_array = process_image(image_path)
    features = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3)).predict(img_array).flatten()
    return features


# Function to make a prediction on a new image
def predict(image_path):
    """Predict the class of the image using the RandomForest model."""
    # Extract features using VGG19
    features = extract_features(image_path)

    # Use the trained Random Forest model to make predictions
    prediction = clf.predict([features])

    # Return the predicted class label
    return prediction[0]  # 'viable' or 'nonviable'


# Example of predicting a new image
image_path = '/Users/nachuthenappan/aihack23/Case-48-P5-C23-29641-13214.jpg'
predicted_class = predict(image_path)

print(f"Predicted class: {predicted_class}")
