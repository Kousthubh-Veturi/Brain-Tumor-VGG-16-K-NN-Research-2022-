import urllib
from urllib import request
import pathlib 
from pathlib import Path
import shutil
import os
from numpy import save
import pathlib
import os
from pathlib import Path
import tensorflow as tf
import keras
from keras import models 
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
import pickle as pkl
def generate_vgg_features():
    '''
    before calling function, define following vars:
    environment variable to access directory for training data
    call environment variable: "MRI_DATA_PATH", points to the dataset
        
    '''

    model = VGG16()
    kept_layers = model.layers[:-1]
    model = keras.Sequential(kept_layers)
  

    for split_name in ["training","testing"]:
        source = Path(os.environ["MRI_DATA_PATH"],split_name)
        class_dirs = sorted(source.glob("[!.]*"))
        for class_dir in class_dirs:
            for file_path in class_dir.iterdir():
                dest = Path("artifacts/vgg_features/",split_name,class_dir.name,file_path.stem).with_suffix(".npy") 
                image = tf.keras.preprocessing.image.load_img(file_path,target_size=(224,224))
                input_arr = preprocess_input(tf.keras.preprocessing.image.img_to_array(image))[np.newaxis,...]
                features = model.predict(input_arr,verbose=0)[0]
                dest.parent.mkdir(parents=True,exist_ok=True)
                np.save(dest,features)
 

def train_nearest_neighbor_classifier():
    features = []
    labels = []
    feature_dir = Path("artifacts/vgg_features/training/")
    for i,class_dir in enumerate(sorted(feature_dir.iterdir())):
         for file_path in class_dir.iterdir():
            features.append(np.load(file_path))
            labels.append(i)


    knc = KNeighborsClassifier(n_neighbors=1)
    knc.fit(features,labels) 
    dest = Path("artifacts/models/knc.pkl")
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(pkl.dumps(knc)) 
    print("saved model") 

def test_nearest_neighbor_classifier():
    features = []
    labels = []
    feature_dir = Path("artifacts/vgg_features/testing/")
    for i,class_dir in enumerate(sorted(feature_dir.iterdir())):
         for file_path in class_dir.iterdir():
            features.append(np.load(file_path))
            labels.append(i)
    labels = np.array(labels)
    features = np.array(features)
    model = pkl.load(open("artifacts/models/knc.pkl", 'rb'))
    predictions = model.predict(features) 
    '''
    model = pkl.load(open("artifacts/models/knc.pkl", 'rb'))
    for feature in features:
        predfeature = np.reshape(feature, (1, 4096))
        result = results.append(model.predict(predfeature))
    '''
    print(np.mean(labels!=predictions)*100 , "%")
    pred_counts = np.zeros((4,4))
    for label,prediction in zip(labels,predictions):
        pred_counts[label,prediction]+=1

    print(pred_counts)
    pred_rates = (pred_counts/(np.sum(pred_counts,axis=1,keepdims=True)))
    print(pred_rates)
    
        


    
