from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import keras.backend as K
import numpy as np
import cv2
import sys

import warnings
warnings.simplefilter('ignore')

nones = lambda n: [None for _ in range(n)]
model, image_path, preprocessed_img, class_output, last_conv_layer, original_img, superimposed_img = nones(7)
def download_pretrained_model():
    global model
    # Initializing our model
    model = VGG16(weights="imagenet")
   
def load_and_preprocess_test_image(img_path):
    global preprocessed_img, image_path

    image_path = img_path
    # The VGG network expects input size to be (224×224×3), so we resize our image to the required size.
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)

    # Since, we are passing only one image through the network,
    # it’s required to expand the first dimension noting it as a batch of size 1.
    x = np.expand_dims(x, axis=0)

    # We then normalize our image using a helper function preprocess_input
    preprocessed_img = preprocess_input(x)

    # Finally, we use OpenCV to read the image,
    global original_img
    original_img = cv2.imread(image_path)

def predict_image_object():
    global class_output, last_conv_layer
    preds = model.predict(preprocessed_img)

    # Seeing the map for the top prediction.
    # So, we take the topmost class index from the predictions for the image.
    # Remember that we can compute map for any class.
    class_idx = np.argmax(preds[0])
    class_output = model.output[:, class_idx]

    # Then, we take the output from the final convolutional layer in the VGG16 which is block5_conv3.
    # The resulting feature map will be of shape 14×14×512.
    last_conv_layer = model.get_layer("block5_conv3")

    top_1 = decode_predictions(preds)[0][0]
    print('Predicted class: %s with probability %.2f' % (top_1[1], 100*top_1[2]))

def apply_grad_cam():
    # We compute the gradient of the class output value with respect to the feature map.
    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # Then, we pool the gradients over all the axes leaving out the channel dimension.
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([preprocessed_img])

    # Finally, we weigh the output feature map with the computed gradient values.
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    # Averaging the weighed feature map along the channel dimension
    # resulting in a heat map of size 14 times 14.
    # And, then we normalize the heat map to make the values in between 0 and 1.
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Resize the existing heatmap to the image size.
    # We blend the original image and the heatmap to superimpose the heatmap on to the image.
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    global superimposed_img
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

def display_image(title):
    import matplotlib.pyplot as plt
    plt.title(title)
    if title == "Original":
        plt.imshow(original_img)
    else:
        plt.imshow(superimposed_img)
    plt.show()
