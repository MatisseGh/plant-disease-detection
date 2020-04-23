import os
import shutil

root = 'C:/Users/matis/Documents/Thesis/deep-learning-for-leafroll-disease-detection/data/'

for folder in os.listdir(root + 'train'):
    images = os.listdir(root + 'train' + '/' + folder)
    val_len = int(0.15*len(images))
    test_len = int(0.15*len(images))
    
    val_dest = 'C:/Users/matis/Documents/Thesis/deep-learning-for-leafroll-disease-detection/data/validation' + '/' + folder
    if not os.path.exists(val_dest):
        os.mkdir(val_dest)
    
    test_dest = 'C:/Users/matis/Documents/Thesis/deep-learning-for-leafroll-disease-detection/data/test' + '/' + folder
    if not os.path.exists(test_dest):
        os.mkdir(test_dest)
    
    for img in images[:val_len]:
        source = root + 'train' + '/' + folder + '/' + img
        shutil.move(source, val_dest)
       
    images = os.listdir(root + 'train' + '/' + folder)
    for img in images[:test_len]:
        source = root + 'train' + '/' + folder + '/' + img
        shutil.move(source, test_dest)

