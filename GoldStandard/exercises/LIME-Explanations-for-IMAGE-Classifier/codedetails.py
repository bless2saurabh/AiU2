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

session, names, probabilities, processed_images, preds, lime_explanation = None, None, None, None, None, None

def load_pretrained_slim_model ():
    global session, names, probabilities, processed_images, slim_home_dir
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

def transform_image_for_inception(imageFilePath):
    image_size = inception.inception_v3.default_image_size
    out = []
    image_raw = tf.image.decode_png(open(imageFilePath, 'rb').read(), channels=3)
    image = inception_preprocessing.preprocess_image(image_raw, image_size, image_size, is_training=False)
    out.append(image)
    return session.run([out])[0][0]

def predict_fn(images):
    global session, names, probabilities, processed_images
    return session.run(probabilities, feed_dict={processed_images: images})

def print_top_predictions_for_image(image, top=5):
    global preds
    preds = predict_fn([image])
    for x in preds.argsort()[0][:-top-1:-1]:
        print(x, names[x], preds[0, x])

def display_image(image):
    plt.imshow(image / 2 + 0.5)
    plt.show()

## LIME for image classsifier
from lime import lime_image
from skimage.segmentation import mark_boundaries

def process_image_by_lime (image):
    global lime_explanation
    lime_explanation = lime_image.LimeImageExplainer().explain_instance(
        image, predict_fn, top_labels=5, hide_color=0, num_samples=1000)

def display_lime_image_exp (top, positive_only, num_features, hide_rest):
    global preds, lime_explanation
    temp, mask = lime_explanation.get_image_and_mask(
        preds.argsort()[0][-top], positive_only, num_features=num_features, hide_rest=hide_rest)
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.show()

def show_lime_explanations_for_object (top):
    display_lime_image_exp(top, positive_only=True, num_features=5, hide_rest=True)

    display_lime_image_exp(top, positive_only=True, num_features=5, hide_rest=False)

    display_lime_image_exp(top, positive_only=False, num_features=10, hide_rest=False)
