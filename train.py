#IMPORTS
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.densenet import DenseNet201
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras import backend as K
import matplotlib.pyplot as plt
import pickle 
import argparse
from pathlib import Path
from collections import Counter


ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", type=str, required=True, choices=['inceptionv3', 'resnet50v2', 'densenet201'], help="Choose model")
ap.add_argument('-t', "--type", type=str, required=True, choices=['scratch', 'deep', 'shallow', 'hybrid'], help="Choose the type of learning")
ap.add_argument("-bs", "--batch_size", type=int, default=32, help="choose batch size")
ap.add_argument("-e", "--epochs", type=int, default=30, help="amount of epochs")
#default is for my server purposes
ap.add_argument("-data_path", "--data_path", type=str, default="/tmp/data/", help="path to images, e.g. './data/'.  Make sure data subfolders is in train/validation/test form")
args = vars(ap.parse_args())

#DEFINE MODELS
MODELS = {'inceptionv3' : InceptionV3, 'resnet50v2' : ResNet50V2, 'densenet201' : DenseNet201}
#DEFINE INPUT SHAPE
INPUTS = {'inceptionv3' : (299,299), 'resnet50v2' : (224,224), 'densenet201' : (224,224)}

#PARAMETERS
dir_root_str = args['data_path']
BATCH_SIZE = args['batch_size']
EPOCHS = args['epochs']
EPOCHS_PRE = 2 #used for hybrid learning
IMG_SIZE = INPUTS[args['model']]
INITIAL_LR = 0.001

#ADD AUGMENTATION 

## Lot of augmenatation for training data
print("[INFO] Defining ImageDataGenerator")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=90,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True, 
    vertical_flip=True,
    brightness_range=[0.5,1.5],
    fill_mode="nearest")

## no further augmentation for validation generator
validation_datagen = ImageDataGenerator(
    rescale=1./255) 

#DEFINE GENERATORS FOR LOADING DATA IN BATCHES
print("[INFO] Loading data")
## Trained set
train_generator = train_datagen.flow_from_directory(
    directory=dir_root_str + 'train/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=42) # set as training data

#ASSIGN CLASS WEIGHTS FOR IMBALANCED DATASETS
counter = Counter(train_generator.classes)                          
max_val = float(max(counter.values()))       
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}
print(class_weights)

#print number of classes
n_classes = len(train_generator.class_indices)
print("[INFO] {} training classes found: ".format(n_classes))

#Validation set
validation_generator = validation_datagen.flow_from_directory(
    directory=dir_root_str + 'validation/', # same directory as training data
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=42) # set as validation data

n_classes_val = len(validation_generator.class_indices)
print("[INFO] {} validation classes found: ".format(n_classes_val))

#LOAD MODEL
print("[INFO] loading {}...".format(args["model"]))
model_name = args['model']
CNN = MODELS[model_name]
training_type = args['type']
base_model = None
##Differentiate between scratch and transfer
if training_type == 'scratch':
    base_model = CNN(weights=None, include_top=False)
else:
    print("[INFO] Downloading {} weights".format(model_name))
    base_model = CNN(weights='imagenet', include_top=False)

#CREATE TRANSFER LEARNING MODEL    
x = base_model.output
# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(x)
# add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# add dropout for regularization
x = Dropout(0.5)(x)
# and a logistic layer with softmax function --number of outputs equals plant_disease classes
predictions = Dense(n_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

#SHALLOW TRAINING
if training_type == 'shallow':
    # only train last custom layers for shallow training
    for layer in base_model.layers:
        layer.trainable = False

#HYBRID TRAINING
if training_type == 'hybrid':
    #Compile model first time (same optimizer)
    model.compile(optimizer=SGD(lr=INITIAL_LR, decay=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics = ['accuracy'])
    #Pre-train for small number of epochs on all layers for deep training
    print("[INFO] Pre-train hybrid")
    model.fit_generator(
        train_generator,
        steps_per_epoch = train_generator.samples // BATCH_SIZE,
        validation_data = validation_generator, 
        validation_steps = validation_generator.samples // BATCH_SIZE,
        class_weight=class_weights,
        epochs = EPOCHS_PRE,
        verbose = 1)
    print("[INFO] Finished pre-training")
    #Set the base_model layers to false for further shallow training
    for layer in base_model.layers:
        layer.trainable = False

#COMPILE MODEL
model.compile(optimizer=SGD(lr=INITIAL_LR, decay=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics = ['accuracy'])


#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=6)


#TRAIN MODEL
print("[INFO] Train model for {} epochs".format(EPOCHS))
history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // BATCH_SIZE,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // BATCH_SIZE,
    class_weight=class_weights,
    epochs = EPOCHS,
    verbose = 1)

#When did earlystopping occur
n_earlystopping = len(history.history['accuracy'])

path_to_save = "./models/{}/{}".format(model_name, training_type)
Path(path_to_save).mkdir(parents=True, exist_ok=True)

#SAVE MODEL
model.save("{}/{}".format(path_to_save, "model.h5"))
print("Saved model to disk")

#PLOT ACCURACY AND LOSS
with open("{}/{}".format(path_to_save, "history_train"), 'wb') as file_pi:
    pickle.dump(history, file_pi)

# Accuracy learning curves
plt.figure(0)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("{}/{}".format(path_to_save, "accuracy.jpg"))
plt.close()

# Loss learning curves
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("{}/{}".format(path_to_save, "loss.jpg"))
plt.close()


with open("{}/{}".format(path_to_save, 'parameters.txt'), 'a') as parameters:
    parameters.write("MODEL {}\n".format(model_name))
    parameters.write("TRAINING TYPE {}\n".format(training_type))
    parameters.write("BATCH_SIZE {}\n".format(BATCH_SIZE))
    parameters.write("EPOCHS {}\n".format(EPOCHS))
    parameters.write("OPTIMIZER_INFO {}\n".format(str(model.optimizer.get_config())))
    if training_type == 'hybrid':
        parameters.write("EPOCHS_PRE {}\n".format(EPOCHS_PRE))

