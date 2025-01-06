import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
import cv2
import joblib
import subprocess

master_folder_path = '/Users/nachuthenappan/aihack23/Training-Set-1'
csv_file_path = '/Users/nachuthenappan/aihack23/ML_Features_1144.csv'
df = pd.read_csv(csv_file_path)

df['classification'] = df['classification'].replace({'non viable': 'nonviable', 'non Viable': 'nonviable', 'Non Viable': 'nonviable'})
df = df[['image.name', 'classification']].dropna()
df['classification'] = df['classification'].apply(lambda x: 'nonviable' if x != 'viable' else 'viable')
df['image.name'] = df['image.name'].apply(lambda x: x.replace(' ', '-') if isinstance(x, str) else x)
df['image.name'] = df['image.name'].apply(lambda x: x if x.endswith('.jpg') else x + '.jpg')

base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def process_image(image_path):
    img_path = os.path.join(master_folder_path, image_path)
    if os.path.exists(img_path):
        try:
            img = cv2.imread(img_path)
            if img is None:
                return None
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            return base_model.predict(img_array).flatten()
        except Exception as e:
            return None
    else:
        return None

features = []
labels = []

for subfolder in os.listdir(master_folder_path):
    subfolder_path = os.path.join(master_folder_path, subfolder)
    if not os.path.isdir(subfolder_path):
        continue
    for image_name in os.listdir(subfolder_path):
        if image_name.endswith('.jpg'):
            image_name_corrected = image_name.replace(' ', '-')
            if not image_name_corrected.endswith('.jpg'):
                image_name_corrected += '.jpg'
            if image_name_corrected in df['image.name'].values:
                label = df[df['image.name'] == image_name_corrected]['classification'].values[0]
                feature = process_image(os.path.join(subfolder, image_name))
                if feature is not None:
                    features.append(feature)
                    labels.append(label)

X = np.array(features)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

joblib.dump(clf, 'random_forest_model.pkl')

subprocess.run(["cp", "random_forest_model.pkl", "/path/to/destination/"])

base_model.save('vgg19_feature_extractor.h5')
subprocess.run(["cp", "vgg19_feature_extractor.h5", "/path/to/destination/"])
