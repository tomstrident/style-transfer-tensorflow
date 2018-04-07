
import os
import imageio
import scipy.misc

import styletransfer as st

def load_image(path, sz):
  
  image = imageio.imread(path)
  image = scipy.misc.imresize(image, sz)

  return image

STYLE_IMAGE_PATH = os.getcwd() + '/Images/vangogh_starry_night.jpg'
CONTENT_IMAGE_PATH = os.getcwd() + '/Images/Tuebingen_Neckarfront.jpg'

#STYLE_IMAGE_PATH = os.getcwd() + '/style-images/starry_night.jpg'
#STYLE_IMAGE_PATH = os.getcwd() + '/style-images/guernica.jpg'
#STYLE_IMAGE_PATH = os.getcwd() + '/style-images/Birds.jpg'
#STYLE_IMAGE_PATH = os.getcwd() + '/style-images/Viva_Espana.jpg'
#STYLE_IMAGE_PATH = os.getcwd() + '/style-images/rain_princess.jpg'
#STYLE_IMAGE_PATH = os.getcwd() + '/style-images/the_shipwreck_of_the_minotaur.jpg'
#STYLE_IMAGE_PATH = os.getcwd() + '/style-images/la_muse.jpg'
#STYLE_IMAGE_PATH = os.getcwd() + '/style-images/Malevich.jpg'
#STYLE_IMAGE_PATH = os.getcwd() + '/style-images/circle.jpg'
#STYLE_IMAGE_PATH = os.getcwd() + '/style-images/miro.jpg'
#STYLE_IMAGE_PATH = os.getcwd() + '/style-images/chagall.jpg'
#STYLE_IMAGE_PATH = os.getcwd() + '/style-images/tree_of_life.jpg'

#CONTENT_IMAGE_PATH = os.getcwd() + '/content-images/hans.jpg'

img_size = [384, 512]

content_image = load_image(CONTENT_IMAGE_PATH, img_size)
style_image = load_image(STYLE_IMAGE_PATH, img_size)
input_image = content_image.copy()

style_transfer = st.StyleTransferModule(content_image, style_image, input_image, num_iters = 1000)
final_image = style_transfer.run()

imageio.imwrite('Output/final.png', final_image)














