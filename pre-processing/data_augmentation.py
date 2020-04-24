# from keras.preprocessing.image import ImageDataGenerator
# import tensorflow as tf
# # Initialising the ImageDataGenerator class. 
# # We will pass in the augmentation parameters in the constructor. 
# datagen = ImageDataGenerator( 
#         rotation_range = 40, 
#         shear_range = 0.2, 
#         zoom_range = 0.2, 
#         horizontal_flip = True, 
#         brightness_range = (0.5, 1.5)) 


# """ TODO boucle for """


# # Loading a sample image  
# img = tf.keras.preprocessing.image.load_img('0.jpg')  
# # Converting the input sample image to an array 
# x = tf.keras.preprocessing.image.img_to_array(img) 
# # Reshaping the input image 
# x = x.reshape((1, ) + x.shape)  
   
# # Generating and saving 5 augmented samples  
# # using the above defined parameters.  
# i = 0
# for batch in datagen.flow(x, batch_size = 1, 
#                           save_to_dir ='preview',  
#                           save_prefix ='image', save_format ='jpeg'): 
#     i += 1
#     if i > 5: 
#         break

# Importing necessary library 
import Augmentor 
# Passing the path of the image directory 
p = Augmentor.Pipeline("data_augm") 
  
# Defining augmentation parameters and generating 5 samples 
p.flip_left_right(0.5) 
p.black_and_white(0.1) 
p.rotate(0.3, 10, 10) 
p.skew(0.4, 0.5) 
p.zoom(probability = 0.2, min_factor = 1.1, max_factor = 1.5) 
p.sample(5) 