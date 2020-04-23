from keras import backend as K
import pickle 
import argparse
import pathlib
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

#PARSE ARGUMENTS
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, choices=['inceptionv3', 'resnet50v2', 'densenet201'])
ap.add_argument("-t", "--type", required=True, choices=['scratch', 'deep', 'shallow','hybrid'])
ap.add_argument("-dp", "--data_path", type=str, default="/tmp/data/", help="path to images, e.g. './data/'.  Make sure data subfolders is in train/validation/test form")
args = vars(ap.parse_args())

#SET INPUT SIZE
INPUTS = {'inceptionv3' : (299,299), 'resnet50v2' : (224,224), 'densenet201' : (224,224)}

#SET VARIABLES
trained_model = args['model']
training_type = args['type']
data_root_str = args['data_path']
IMG_SIZE = INPUTS[args['model']]


#CHECK IF MODEL.H5 FILE EXISTS
model_path_dir = "./models/{}/{}".format(trained_model, training_type)

file = pathlib.Path("{}/model.h5".format(model_path_dir))
if file.exists ():
    print ("File exist")
else:
    print ("File doesn't exist")


#LOAD MODEL
print("[INFO] Loading model")
model = load_model("{}/model.h5".format(model_path_dir))
print("[INFO] Finished loading model")

#RESCALE DATA
test_datagen = ImageDataGenerator(rescale = 1./255)

#LOAD DATA
print("[INFO] LOADING DATA")
test_generator = test_datagen.flow_from_directory(
    directory=data_root_str + 'test/',
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode='categorical',
    shuffle=False,
    seed=42)


#TESTING MODEL
print("[INFO] Evaluate generator: Testing model {} with training type {}".format(trained_model, training_type))
history = model.evaluate_generator(test_generator, verbose=1)
print("[INFO] Finished evaluating model")

print("[INFO] Writing history object to {}".format(model_path_dir + '/history_test'))
with open("{}/history_test".format(model_path_dir), 'wb') as file:
	pickle.dump(history, file)

print("[INFO] Predict generator: Testing model {} with training type {}".format(trained_model, training_type))
Y_pred = model.predict_generator(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

print("[INFO] Creating classification report")
from sklearn.metrics import classification_report

true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys()) 
report = classification_report(true_classes, y_pred, target_names=class_labels, output_dict=True)

with open("{}/classification_report.txt".format(model_path_dir), 'w') as fd:
    for key in report:
        fd.write("{}: {}\n".format(key, str(report[key])))

print("[INFO] Append result to results.csv")
with open('results.csv','a') as fd:
    fd.write("{},{},{},{}\n".format(trained_model, training_type, history[1], report['weighted avg']['f1-score']))

print("FINISHED")

