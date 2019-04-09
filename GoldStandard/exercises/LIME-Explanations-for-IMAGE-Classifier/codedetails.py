import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
slim = tf.contrib.slim

import sys, os
slim_home_dir = os.path.join('.', 'tf-models', 'slim')
sys.path.append(slim_home_dir)
sys.path.append('.')

from nets import inception
from preprocessing import inception_preprocessing
from datasets import imagenet

session, names, probabilities, processed_images = None, None, None, None


def load_pretrained_model_from_disk(slim_home_dir):
    global session, names, probabilities, processed_images
    names = imagenet.create_readable_names_for_imagenet_labels()
    processed_images = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, _ = inception.inception_v3(processed_images, num_classes=1001, is_training=False)
    probabilities = tf.nn.softmax(logits)
    checkpoints_dir = os.path.join(slim_home_dir, 'pretrained')
    session = tf.Session()
    init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, 'inception_v3.ckpt'),
                                             slim.get_model_variables('InceptionV3'))
    init_fn(session)


def transform_img_fn(relativePathtoImageFile, imageFilename):
    imageFilePath = os.path.join(relativePathtoImageFile, imageFilename)
    image_size = inception.inception_v3.default_image_size
    out = []
    image_raw = tf.image.decode_png(open(imageFilePath, 'rb').read(), channels=3)
    image = inception_preprocessing.preprocess_image(image_raw, image_size, image_size, is_training=False)
    out.append(image)
    return session.run([out])[0][0]


def predict_fn(images):
    global session, names, probabilities, processed_images
    return session.run(probabilities, feed_dict={processed_images: images})


def get_prediction_for(image, top=5):
    preds = predict_fn([image])
    for x in preds.argsort()[0][-top:]:
        print(x, names[x], preds[0, x])


def displayImage(image):
    plt.imshow(image / 2 + 0.5)
    plt.show()