
import scipy.io

import numpy as np
import tensorflow as tf

class StyleTransferModule:
  
  def __init__(self, content_image, style_image, input_image, alpha=1, beta=1, 
               num_iters=100, model_path='Models/imagenet-vgg-verydeep-19.mat'):
    
    self.imgnet_mean = np.array([123.68, 116.779, 103.939])

    self.content_image = self.prep_image(content_image)
    self.style_image = self.prep_image(style_image)
    self.input_image = self.prep_image(input_image)

    self.alpha = alpha
    self.beta = beta
    self.num_iters = num_iters

    self.num_layers = 36
    self.vgg_data = self.load_network_data(model_path)
  
  def prep_image(self, image):
    
    image = image - self.imgnet_mean
    image = image[...,::-1] #BGR
    image = np.reshape(image, ((1,) + image.shape))
    
    return np.float32(image)
  
  def postp_image(self, image):
    
    image = image[0,:,:,::-1] #RGB
    image = image + self.imgnet_mean
    image = np.clip(image, 0, 255).astype('uint8')
    
    return image
    
  def load_network_data(self, model_path):
    
    layers = []
    weights = []
    biases = []
    
    data = scipy.io.loadmat(model_path)
    network_data = data['layers']
    
    for i in range(self.num_layers):
      name = network_data[0][i][0][0][0][0]
      if name[:4] == 'conv':
        w = network_data[0][i][0][0][2][0][0]
        b = network_data[0][i][0][0][2][0][1]
        weights.append(w.transpose((1, 0, 2, 3)))
        biases.append(b.reshape(-1))
      layers.append(name)
    
    return [layers, weights, biases] 
  
  # input image: [batch, height, width, channels]
  # weights: [height, width, channels_in, channels_o]
  
  def vgg(self, vgg_data, input_image):
    
    layers = vgg_data[0]
    weights = vgg_data[1]
    biases = vgg_data[2]
    
    idx = 0
    net = {}
    node = input_image
      
    for layer in layers:
      name = layer[:4]
      if name == 'conv':
        w = weights[idx]
        b = biases[idx]
        idx += 1
        node = tf.nn.bias_add(tf.nn.conv2d(node, tf.constant(w), strides=(1, 1, 1, 1), padding='SAME'), b)
      elif name == 'relu':
        node = tf.nn.relu(node)
      elif name == 'pool':
        node = tf.nn.max_pool(node, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
      net[layer] = node
  
    return net
    
  def gram_matrix(self, mat):
    
    b, h, w, c = mat.get_shape()
    F = tf.reshape(mat, (h*w, c))
    G = tf.matmul(tf.transpose(F), F) / int(h*w)
    
    return G
    
  def run(self):
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      
      x = tf.Variable(self.input_image, trainable=True, dtype=tf.float32)
      
      p = tf.placeholder(tf.float32, shape=self.content_image.shape)
      a = tf.placeholder(tf.float32, shape=self.style_image.shape)
      
      content_layer = ['relu4_2']
      style_layer = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
      
      # content layer
      P = self.vgg(self.vgg_data, p)
      P = [P[l] for l in content_layer]
      
      # style layer
      A = self.vgg(self.vgg_data, a)
      A = [self.gram_matrix(A[l]) for l in style_layer]
      
      # input layer
      X = self.vgg(self.vgg_data, x)
      F = [X[l] for l in (content_layer)]
      G = [self.gram_matrix(X[l]) for l in (style_layer)]
      
      content_weights = [1e0]
      style_weights = [1e3/n**2 for n in [64,128,256,512,512]]
      
      L_content = 0.0
      L_style = 0.0
      
      # content loss
      for l, n in enumerate(content_layer):
        L_content += content_weights[l] * tf.reduce_mean((F[l] - P[l]) ** 2.0)
      
      # style loss
      for l, n in enumerate(style_layer):
        L_style += style_weights[l] * tf.reduce_mean((G[l] - A[l]) ** 2.0)
      
      # total loss
      L_total = self.alpha*L_content + self.beta*L_style
    
      global it
      it = 0
      
      def callback(t_loss, c_loss, s_loss):
        global it
        print('iteration: %4d' % it, 
              'total: %12g, content: %12g, style: %12g' % (t_loss, c_loss, s_loss))
        it += 1
      
      sess.run(tf.global_variables_initializer())
      
      optimizer = tf.contrib.opt.ScipyOptimizerInterface(L_total, 
                                                         method='L-BFGS-B', 
                                                         options={'maxiter': self.num_iters})
      optimizer.minimize(sess, 
                         feed_dict={p:self.content_image, a:self.style_image}, 
                         fetches=[L_total, L_content, L_style], 
                         loss_callback=callback)
      
      final_image = sess.run(x)
    return self.postp_image(final_image)
      
    




