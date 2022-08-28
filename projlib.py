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
    feature_dir = Path("artifacts/vgg_features/training/")
    for path in feature_dir.iterdir():
        features.append(np.load(path))
    
    features = np.array(features)
    #need labels, how get labels
    sample, x, y = np.shape(features)
    shapedarray = np.reshape(features,(sample,x*y))

    nnc = NearestNeighbors(n_neighbors=4)
    nnc.fit(shapedarray)

    dest = Path("artifacts/models/")
    dest.mkdir(parents=True,exist_ok=True)
    pickle.dump(nnc,open("artifacts/models/nearest_neighbor_classifier.pkl","wb"))
    print("saved model")
