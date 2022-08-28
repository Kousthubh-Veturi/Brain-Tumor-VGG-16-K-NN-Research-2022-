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
def download_dataset():
    #kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
    remote_url = 'https://storage.googleapis.com/kaggle-data-sets/1608934/2645886/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220718%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220718T200302Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=67bc45750658fff5168c600bf08014cca5abd6df4836296f9b6d7754c586cb57caa931adbdd866d5a45f2fb51e19bb78c7839d09c9f782074c10d90d576522fc77ad35e17395c132db2611e9d5b5aada212e1163d7d08c53b1c4724b68bd22c28758f68b11ea832b7099c921c4d2f60141ac57c8a6d1c3adb3115c307323c5851faefa4123efc1ea77ef601cbf515b8d44beaea2764948e7a4483f23c0066c26b6cf42eb5e41800987e1d7b24f2a8ba892cbdf34be0a0ff9d38b354ed89df62bb7c0c8f27ad7a4ff60be76857af7fbcc20b0f6eef3c017c1ca60c907336afdbe46ac89743a563de614b9d9edbd12529e7b1d3ec02c46202e16e2e56cecb32d28'
    #remote_url = 'https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/download?datasetVersionNumber=1'
    #remote_url = 'https://drive.google.com/file/d/17BbMYOheRyMSh7yxBALkd0k0lxAIzSzQ/view?usp=sharing'
    local_file = Path('artifacts/dataset.zip')
    local_file.parent.mkdir(exist_ok=True)
    request.urlretrieve(remote_url,local_file)
    shutil.unpack_archive(local_file,'artifacts/dataset')


def generate_vgg_features():
    '''
    before calling function, define following vars:
    environment variable to access directory for training data
    call environment variable: "MRI_DATA_PATH", points to the dataset
        
    '''

    model = VGG16()
    kept_layers = model.layers[:-1]
    model = keras.Sequential(kept_layers)
  

    for split in ["training","testing"]:
        source = Path(os.environ["MRI_DATA_PATH"],split) 
        images = tf.keras.utils.image_dataset_from_directory(source,batch_size=1,image_size=(224,224))
        dest = Path("artifacts/vgg_features/",split)
        dest.mkdir(parents=True,exist_ok=True)
        for i,(image,_) in enumerate(images):
            image = preprocess_input(image)
            features = model.predict(image,verbose=0) 
            np.save(f"{dest}/{i:08d}.npy",features[0])




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
