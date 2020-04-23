from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
# load the image
img = load_img('C:/Users/matis/Documents/Thesis/deep-learning-for-leafroll-disease-detection/utils/images/grape_black_measel_original.JPG')
# convert to numpy array
data = img_to_array(img)
# expand dimension to one sample
samples = expand_dims(data, 0)
# create image data augmentation generator
datagen = ImageDataGenerator(width_shift_range=[0.3,0.4])
# prepare iterator
it = datagen.flow(samples, batch_size=1)


# define subplot
pyplot.subplot(330 + 1)
# generate batch of images
batch = it.next()
# convert to unsigned integers for viewing
image = batch[0].astype('uint8')
# plot raw pixel data
pyplot.imsave('width_shift.JPG', image)
# show the figure
pyplot.show()

datagen = ImageDataGenerator(height_shift_range=[0.3,0.4])
# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot
# define subplot
pyplot.subplot(330 + 1)
# generate batch of images
batch = it.next()
# convert to unsigned integers for viewing
image = batch[0].astype('uint8')
# plot raw pixel data
pyplot.imsave('height_shift.JPG', image)
# show the figure
pyplot.show()


datagen = ImageDataGenerator(horizontal_flip=True)
# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot

# define subplot
pyplot.subplot(330 + 1)
# generate batch of images
batch = it.next()
# convert to unsigned integers for viewing
image = batch[0].astype('uint8')
# plot raw pixel data
pyplot.imsave('horizontal_flip.JPG',image)
# show the figure
pyplot.show()


datagen = ImageDataGenerator(vertical_flip=True)
# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot

# define subplot
pyplot.subplot(330 + 1)
# generate batch of images
batch = it.next()
# convert to unsigned integers for viewing
image = batch[0].astype('uint8')
# plot raw pixel data
pyplot.imsave('vertical_flip.JPG',image)
# show the figure
pyplot.show()
datagen = ImageDataGenerator(rotation_range=90)
# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot

# define subplot
pyplot.subplot(330 + 1)
# generate batch of images
batch = it.next()
# convert to unsigned integers for viewing
image = batch[0].astype('uint8')
# plot raw pixel data
pyplot.imsave('rotation.JPG',image)
# show the figure
pyplot.show()

datagen = ImageDataGenerator(brightness_range=[1.5,1.6])
# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot

# define subplot
pyplot.subplot(330 + 1)
# generate batch of images
batch = it.next()
# convert to unsigned integers for viewing
image = batch[0].astype('uint8')
# plot raw pixel data
pyplot.imsave('brightess.JPG',image)
# show the figure
pyplot.show()

datagen = ImageDataGenerator(zoom_range=[0.5,0.7])
# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot

# define subplot
pyplot.subplot(330 + 1)
# generate batch of images
batch = it.next()
# convert to unsigned integers for viewing
image = batch[0].astype('uint8')
# plot raw pixel data
pyplot.imsave('zoom.JPG',image)
# show the figure
pyplot.show()
