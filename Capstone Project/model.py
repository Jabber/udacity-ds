from keras import backend
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Sequential
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from tqdm import tqdm
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
import urllib
from io import BytesIO
from skimage import io

# extract pre-trained face detector
dog_names = pickle.load(open("dog_names.pickle", "rb"))
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# helper to avoid image download
def loadImage(URL):
    with urllib.request.urlopen(URL) as url:
        img = image.load_img(BytesIO(url.read()), target_size=(224, 224))
        img = image.img_to_array(img)
    return img

# helper
def path_to_tensor(img_path):
    img = loadImage(img_path)
    return np.expand_dims(img, axis=0)

# helper
def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

# helper
def extract_Resnet50(tensor):
    from keras.applications.resnet50 import ResNet50, preprocess_input
    return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

# returns predicted breed name
def Resnet50_predict_breed(img_path, model):
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    predicted_vector = model.predict(bottleneck_feature)
    return dog_names[np.argmax(predicted_vector)]


# used for breed prediction
def ResNet50_predict_labels(img_path):
    ResNet50_model = ResNet50(weights='imagenet')
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


# returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))


# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = loadImage(img_path).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


# returns error if there is no human/dog, otherwise finds the most similar dog breed
def dog_predict(img_path):
    backend.clear_session()
    model = Sequential()
    model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
    model.add(Dense(133, activation='softmax'))
    model.load_weights('saved_models/weights.best.Resnet50.hdf5')

    if dog_detector(img_path) or face_detector(img_path):
        return Resnet50_predict_breed(img_path, model).split('.')[1]
    else:
        return "No human/dog found on image"
