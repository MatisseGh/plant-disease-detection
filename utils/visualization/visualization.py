from keras.applications import ResNet50
from vis.utils import utils
from vis.visualization import visualize_saliency, overlay
from keras import activations
from glob import glob
import argparse
import re
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import numpy as np
from matplotlib import pyplot as plt
import os


ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", type=str, required=True, choices=['inceptionv3', 'resnet50v2', 'densenet201'], help="Choose model")
ap.add_argument('-t', "--type", type=str, required=True, choices=['scratch', 'deep', 'shallow', 'hybrid'], help="Choose the type of learning")
args = vars(ap.parse_args())

class_names = glob("./data/train/*") # Reads all the folders in which images are present
for i in range(len(class_names)):
	class_names[i] = re.sub(r'.*\\', '', class_names[i])
class_names = sorted(class_names) # Sorting them
name_id_map = dict(zip(class_names, range(len(class_names))))

model = load_model("./models/{}/{}/model.h5".format(args['model'], args['type']))

img_path = './Peach bacterial spot.JPG'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

filter_index = name_id_map['Peach___Bacterial_spot']

layer_idx = -1

model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

grads = visualize_saliency(model, layer_idx, filter_indices=filter_index, seed_input=x, backprop_modifier='guided')
    
plt.imshow(img)
plt.imshow(grads, alpha=0.3)
plt.axis('off')
plt.imshow(grads)
plt.savefig('vis.png')
