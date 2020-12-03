import numpy as np 
from keras.preprocessing.image import load_img, img_to_array, array_to_img 
from keras import backend as K
from keras.applications.vgg16 import preprocess_input, VGG16
from IPython.display import display 
from keras.optimizers import SGD, Adam 

content_image_path = 'input/picture_vgg.png'
style_image_path = 'input/KNNstyle.jpg'

w, h = load_img(content_image_path).size 

height = 512 
width = int(w * height / h) 

def preprocess_image(image_path): 
    image = load_img(path = image_path, target_size = (width, height)) 
    image = img_to_array(image) 

    image = np.array([image]) 
    image = preprocess_input(image)
    return image 

content_image = preprocess_image(content_image_path) 
style_image = preprocess_image(style_image_path) 

def postprocess_array(x): 
    t = x.copy() 
    t = t.reshape((width, height, 3)) 
    vgg_mean = [103.939, 116.779, 123.68] 
    for i in range(3): 
        t[:, :, i] += vgg_mean[i] 
    t = t[:, :, ::-1]
    t = np.clip(t, 0, 255).astype('uint8') 
    return t

content_input = K.constant(content_image) 
style_input = K.constant(style_image) 
output_image = K.variable(content_image) 

input_tensor = K.concatenate([content_input, style_input, output_image], axis = 0) 

# import tensorflow as tf 
# import keras 

# print("input_tensor.type: ", type(input_tensor))
# input_tensor = tf.convert_to_tensor(input_tensor)
# print("input_tensor.type: ", type(input_tensor))
# w = keras.Input(tensor = input_tensor)
print('Start load model...') 
model = VGG16(input_tensor = input_tensor, weights='input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)
print('Model has been load.') 

output_dict = dict([(layer.name, layer.output) for layer in model.layers]) 

model.summary() 
print('Model.input: ', model.input) 
# print(output_dict) 
content_weight = 0.5 
style_weight = 1e4 

def get_content_loss(content_features, output_features): 
    return 0.5 * K.sum(K.square(output_features - content_features)) 

layer_feat = output_dict['block5_conv3'] 
content_feat = layer_feat[0, :, :, :] 
output_feat = layer_feat[2, :, :, :] 
loss = content_weight * get_content_loss(content_feat, output_feat)

print("loss = ", loss)
def get_gram_matrix(x): 
    feature_matrix = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1))) 
    gram_matrix = K.dot(feature_matrix, K.transpose(feature_matrix)) 
    return gram_matrix 

def get_style_loss(style_features, output_features): 
    G = get_gram_matrix(style_features) 
    A = get_gram_matrix(output_features) 

    channel_number = int(style_features.shape[2]) 
    size = int(style_features.shape[0]) * int(style_features.shape[1]) 
    return K.sum(K.square(G -A)) / (4.0 * (channel_number ** 2) * (size ** 2))

layer_names = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'] 

for layer_name in layer_names: 
    layer_feat = output_dict[layer_name] 
    style_feat = layer_feat[1, :, :, :] 
    output_feat = layer_feat[2, :, :, :] 
    single_style_loss = get_style_loss(style_feat, output_feat) 
    loss += (style_weight / len(layer_names)) * single_style_loss 
