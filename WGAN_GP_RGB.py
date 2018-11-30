from __future__ import print_function, division
import tarfile
import numpy as np
from matplotlib import pyplot as plt
from io import BytesIO
import cv2
from PIL import Image

from keras.layers.merge import _Merge
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose, MaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras import layers
import matplotlib.pyplot as plt
from functools import partial
import keras.backend as K

class pic(): # Define a Class to do some pre operations of images 
  def __init__(self, filename):
    self.filename = filename;
    self.width = 120 # image width
    self.height = 48 # image height
    self.name_list, self.file_list = self.untar(self.filename)
    self.data_set = self.getdataset(self.file_list)
  
  def untar(self, filename): # uncompress the tar.gz file
    name_list = []
    file_list = []
    with tarfile.open(filename, "r") as file:
        for i in file.getmembers(): # use tar file module to untar the file
            f = file.extractfile(i)
            if f is not None:
              content = f.read()
              name_list.append(i.name)
              file_list.append(content)
    
    return name_list, file_list # return the file name list and file list

  def getdataset(self, file_list): # get data_set with np.array format
    data_set = []
    for i in range(len(file_list)):
      try:
        im = Image.open(BytesIO(file_list[i]))  # file_list is the binary stream, that needs to transfer to image format
      except OSError:
        pass
      
      im = im.convert("RGB") # convert images to RGB scale
      image = np.asarray(im)
      left, right = self.splitpic(image)
      data_set.append(left)
      data_set.append(right)
    return np.array(data_set)

  def showim(self,picture): # show image
    plt.imshow(picture)
    plt.show()

  def norm_img(self,img):
    img = (img / 127.5) - 1
    return img
  
  def denorm(self,img):
    img = (img + 1) * 127.5
    return img.astype(np.uint8) 
  
  def splitpic(self,picture): # split the image
    return np.split(picture, 2, axis=1)[0], np.split(picture, 2, axis=1)[1]
  
  def gendataset(self): # return the data_set
    return self.data_set

class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        e = K.random_uniform((32, 1, 1, 1))
        return (e * inputs[0]) + ((1 - e) * inputs[1])

class WGAN_GP():
    def __init__(self):
        self.img_rows = 48
        self.img_cols = 60
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        self.n_ = 5 # paper suggested 
        optimizer = RMSprop(lr=0.00005) # paper suggested 
        
        self.generator = self._generator()
        self.discriminator = self._discriminator()

        # Freeze generator
        self.generator.trainable = False

        # real sample
        real_img = Input(shape=self.img_shape)
        # Generate random noise
        noise = Input(shape=(self.latent_dim,))
        # fake image
        fake_img = self.generator(noise)

        fake = self.discriminator(fake_img)
        valid = self.discriminator(real_img)

        # After RandomWeightedAverage could get a interpolated image x1*e + (1-e)x2
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        
        # Sample image
        validity_interpolated = self.discriminator(interpolated_img)

        partial_gp_loss = partial(self.gradient_penalty_loss, averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty'

        self.discriminator_model = Model(inputs=[real_img, noise], outputs=[valid, fake, validity_interpolated])
        
        # we apply three loss functions for each dimension
        self.discriminator_model.compile(loss=[self.wasserstein_loss, self.wasserstein_loss, partial_gp_loss], optimizer=optimizer, loss_weights=[1, 1, 10])

        # Freeze discriminator for training generator
        self.discriminator.trainable = False
        self.generator.trainable = True

        generated = Input(shape=(100,))
        img = self.generator(generated)
        valid = self.discriminator(img)

        self.generator_model = Model(generated, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)
        
        
    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)
      
    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples): #The process is to compute the square root of the sum of squares of gradient D(x)
        gradients = K.gradients(y_pred, averaged_samples)[0]
        gradients_sqr = K.square(gradients)
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        gradient_penalty = K.square(1 - gradient_l2_norm)
        return K.mean(gradient_penalty)  
      
    def _generator(self):


        model = Sequential()

        model.add(Dense(128 * 12 * 15, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((12, 15, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def _discriminator(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, data_set, batch_size=128, save_interval=50):
        
        data_set = data_set / 127.5 - 1. # Rescale -1 to 1
        
        real = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        sample = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):

            for _ in range(self.n_):

              idx = np.random.randint(0, data_set.shape[0], batch_size)
              real_images = data_set[idx]

              noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

              d_loss = self.discriminator_model.train_on_batch([real_images, noise],[real, fake, sample])

            g_loss = self.generator_model.train_on_batch(noise, real)

            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))
              
            if epoch % save_interval == 0:
                self.show_imgs(epoch)

    def show_imgs(self, epoch):
        # Generate 3*3
        noise = np.random.normal(0, 1, (3 * 3, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
         
        gen_imgs = (gen_imgs + 1) * 127.5
        
        fig, axs = plt.subplots(3, 3)
        current = 0
        for i in range(3):
            for j in range(3):
                axs[i,j].imshow(gen_imgs[current].astype(np.uint8))
                axs[i,j].axis('off')
                current += 1
        plt.show()
        plt.close()
    
    def dataset(self, data_set):
      data_set = self.getdataset(file_list, 48, 120)
      return data_set

pic1 = pic('yzm1.tar.gz')
data_set = pic1.gendataset()
print(data_set.shape)

if __name__ == '__main__':
    dcgan = WGAN_GP()
    dcgan.train(epochs=25000, batch_size=32, save_interval=50, data_set = data_set)

