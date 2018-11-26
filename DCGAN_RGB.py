from __future__ import print_function, division
import tarfile
import numpy as np
from matplotlib import pyplot as plt
from io import BytesIO
import cv2
from PIL import Image

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose, MaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

pic1 = pic('yzm1.tar.gz')
data_set1 = pic1.gendataset()
print (data_set1.shape)

class DCGAN():
    def __init__(self):
        self.img_height = 48
        self.img_width = 60
        self.channels = 3
        self.img_shape = (self.img_height, self.img_width, self.channels)
        self.latent_dim = 100

        # Define optimizer
        optimizer = Adam(0.0002, 0.5)

        # Build the generator
        self.generator = self._generator()
        
        # Build discrimnator
        self.discriminator = self._discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Noise as input
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # In order to train G, we need to fix D
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        real = self.discriminator(img)

        # combine the generator and discriminator
        # Trains the generator to fool the discriminator
        self.combined = Model(z, real)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def _generator(self):

        moment = 0.5
        model = Sequential()

        model.add(Dense(256 * 12 * 15, activation="relu", input_dim=self.latent_dim))
        model.add(BatchNormalization(momentum=moment))
        model.add(LeakyReLU(0.2))
        model.add(Reshape((12, 15, 256)))
        
        model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=moment))
        model.add(LeakyReLU(0.2))
        
        model.add(Conv2DTranspose(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=moment))
        model.add(LeakyReLU(0.2))
        
        model.add(Conv2DTranspose(32, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=moment))
        model.add(LeakyReLU(0.2))
        
        model.add(Conv2DTranspose(16, kernel_size=3, padding='same'))
        model.add(BatchNormalization(momentum=moment))
        model.add(LeakyReLU(0.2))
        
        model.add(Conv2DTranspose(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def _discriminator(self):

        dropout = 0.3
        
        model = Sequential()

        model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))
        
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8)) # Batchnormalzation learing param
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))
        
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))
        
        model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))
        
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        
        model.summary()
        img = Input(shape=self.img_shape)
        
        validity = model(img)
        
        return Model(img, validity)

    def train(self, epochs, data_set, batch_size=128, save_interval=50):
        
        data_set = data_set / 127.5 - 1. # Rescale -1 to 1
        
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        start = 0
        for epoch in range(epochs):

            # Train D
            self.generator.trainable = False
            self.discriminator.trainable = True
            #idx = np.random.randint(0, data_set.shape[0], batch_size)
            #imgs = data_set[idx]
            end = start + batch_size
            real_images = data_set[start:end]

            # Noise as input and get generated image 
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(real_images, real)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train G
            self.discriminator.trainable = False
            self.generator.trainable = True
            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, real)

            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))
            
            start += batch_size
            if start > len(data_set) - batch_size:
              start = 0
              
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

if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=20000, batch_size=32, save_interval=50, data_set = data_set1)