import tensorflow as tf
import numpy as np
import os 
import json
from sklearn.utils import shuffle
from tqdm import tqdm

tf.enable_eager_execution()

def load_image_inception(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

def initialize_inception_model():
  image_model = tf.keras.applications.InceptionV3(include_top=False, 
                                                weights='imagenet')
  new_input = image_model.input
  hidden_layer = image_model.layers[-1].output

  image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
  return image_features_extract_model

image_features_extract_model = initialize_inception_model()


annotation_file = 'annotations/captions_train2014.json'
# read the json file
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# storing the captions and the image name in vectors
all_captions = []# only has the captions
all_img_name_vector = []# only has the paths

for annot in annotations['annotations']:
    #have to see if each image has 1:5 relation with the captions and have to incorporate that somehow!
    caption = '<start> ' + annot['caption'] + ' <end>'
    image_id = annot['image_id']
    PATH = 'content/train2014/'
    full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)
    
    all_img_name_vector.append(full_coco_image_path)
    all_captions.append(caption)

# shuffling the captions and image_names together
# setting a random state
train_captions, img_name_vector = shuffle(all_captions,
                                          all_img_name_vector,
                                          random_state=1)

print('_'*50)
print("\tTotal No. of Images   : ", len(img_name_vector))
print("\tTotal No. of Captions : ", len(train_captions))
print('_'*50)

img_name_test = np.load('MS_COCO/img_name_test.npy')
cap_test = np.load('MS_COCO/cap_test.npy')

print('_'*50)
print("\tTotal No. of Test Images   : ", img_name_test.shape)
print('_'*50)

img_name_test = img_name_test[:6000]
cap_test = cap_test[:6000]

print('_'*50)
print("\tTotal No. of Test Images   : ", img_name_test.shape)
print('_'*50)


img_name_test = list(map(lambda x: x[1:], img_name_test))

# getting the unique images
encode_train = sorted(set(img_name_test))

# feel free to change the batch_size according to your system configuration
# a generator which yields 16 images at one shot along with its path
image_dataset = tf.data.Dataset.from_tensor_slices(
                                encode_train).map(load_image_inception).batch(16)

for img, path in image_dataset:
  #image_features_extract_model is the InceptionV3/Resnet stack which gives out tensors in the end 
  batch_features = image_features_extract_model(img)
  batch_features = tf.reshape(batch_features, 
                              (batch_features.shape[0], -1, batch_features.shape[3]))
  

  for bf, p in zip(batch_features, path):
    path_of_feature = p.numpy().decode("utf-8")
    np.save(path_of_feature, bf.numpy())    