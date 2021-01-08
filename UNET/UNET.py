import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import merge, Input, concatenate, Dropout, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, UpSampling2D, UpSampling3D, Cropping3D, BatchNormalization, Activation, add

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.utils import plot_model

import tensorflow as tf
import vtk
import glob
import pickle
import operator
import csv

import scipy
from scipy import ndimage

import time

class myUnet(object):

    def __init__(self):
        self.working_folder = 'D:/'
        self.test_folder = self.working_folder+"/test"
        self.train_folder = self.working_folder+"/train"        
        self.validation_folder = self.working_folder+"/validation"
        self.result_folder = self.working_folder+"/result"
        self.predict_folder = self.test_folder+"/predict"
        self.num_channels = 4
    
    def normalize_image(self,im):
        avg = np.mean(im,dtype='float32')
        std = np.std(im,dtype='float32')
        return (im-avg)/std
    
    def load_data(self):
        
        self.imgs_train = self.load_data_2d(os.path.join(self.train_folder,'t1.npy'))
        self.imgs_train = np.append(self.imgs_train,self.load_data_2d(os.path.join(self.train_folder,'t1ce.npy')),axis=3)
        self.imgs_train = np.append(self.imgs_train,self.load_data_2d(os.path.join(self.train_folder,'t2.npy')),axis=3)
        self.imgs_train = np.append(self.imgs_train,self.load_data_2d(os.path.join(self.train_folder,'flair.npy')),axis=3)

        self.imgs_mask_train = self.load_data_2d(os.path.join(self.train_folder,'seg.npy'),False)
        print('Total_imgs_shape:',self.imgs_train.shape)

#        edema is 2, non-enhanced tumor is 1, enhanced tumor is 4 
#           use following lines to decide which contour to train on


#        self.imgs_mask_test[self.imgs_mask_test>0] = 1
#        self.imgs_mask_validation[self.imgs_mask_validation>0] = 1
        self.imgs_mask_train[self.imgs_mask_train>0] = 1
        
#        self.imgs_mask_test[self.imgs_mask_test==2] = 0
#        self.imgs_mask_train[self.imgs_mask_train==2] = 0
#        self.imgs_mask_test[self.imgs_mask_test==4] = 1
#        self.imgs_mask_train[self.imgs_mask_train==4] = 1
        
#        self.imgs_mask_test[self.imgs_mask_test>1] = 0
#        self.imgs_mask_train[self.imgs_mask_train>1] = 0
        
        self.img_rows = self.imgs_train.shape[1]
        self.img_cols = self.imgs_train.shape[2] 
        print('mask_highest_number:',np.max(self.imgs_mask_train))
        self.num_channels = 4
    
    def load_data_2d(self,img_file,normalize = True, flip = False):
        imgs = np.load(img_file)

        imgs = np.swapaxes(np.swapaxes(imgs,1,3),2,3)
            
        img_rows = imgs.shape[2]
        print('imgs_rows:', img_rows)  
        img_cols = imgs.shape[3]
        print('imgs_cols:', img_cols)  
        count = imgs.shape[0]*imgs.shape[1]
        print('imgs_count:', count)  

        imgs=np.reshape(imgs,[count,img_rows,img_cols,1])
        print('imgs_shape:', imgs.shape)        
        if normalize:
            imgs = imgs.astype(np.float32)
            for i in range(0,count):
             
                imgs[i,:,:,0] = self.normalize_image(imgs[i,:,:,0]) 
               
            imgs = imgs.astype(np.float16)
                
        if flip:               
            flip_imgs = imgs
            for i in range(0,count):
                flip_imgs[i,:,:,0] = np.flip(imgs[i,:,:,0],axis=0)
            return np.append(imgs,flip_imgs,axis=0)
        
        return imgs
    
       


    def get_unet_2d_bn(self,feature_base = 64):

        inputs = Input((self.img_rows, self.img_cols,self.num_channels))
        conv1 = Conv2D(feature_base, 3, activation = 'relu', padding = 'same', kernel_initializer = 'Orthogonal')(inputs)
        print ("conv1 shape:",conv1.shape)
        conv1 = Conv2D(feature_base, 3, activation = 'relu', padding = 'same', kernel_initializer = 'Orthogonal')(conv1)
        print ("conv1 shape:",conv1.shape)        
        bn1 = BatchNormalization()(conv1)
        conv_atv1 = Activation('relu')(bn1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv_atv1)
        print("pool1 shape:",pool1.shape)

        conv2 = Conv2D(feature_base*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'Orthogonal')(pool1)
        print("conv2 shape:",conv2.shape)
        conv2 = Conv2D(feature_base*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'Orthogonal')(conv2)
        print("conv2 shape:",conv2.shape)      
        bn2 = BatchNormalization()(conv2)
        conv_atv2 = Activation('relu')(bn2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv_atv2)
        print("pool2 shape:",pool2.shape)

        conv3 = Conv2D(feature_base*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'Orthogonal')(pool2)
        print("conv3 shape:",conv3.shape)
        conv3 = Conv2D(feature_base*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'Orthogonal')(conv3)
        print("conv3 shape:",conv3.shape)      
        bn3 = BatchNormalization()(conv3)
        conv_atv3 = Activation('relu')(bn3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv_atv3)
        print("pool3 shape:",pool3.shape)

        conv4 = Conv2D(feature_base*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'Orthogonal')(pool3)
        conv4 = Conv2D(feature_base*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'Orthogonal')(conv4)      
        bn4 = BatchNormalization()(conv4)
        conv_atv4 = Activation('relu')(bn4)
        drop4 = Dropout(0.5)(conv_atv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(feature_base*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'Orthogonal')(pool4)
        conv5 = Conv2D(feature_base*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'Orthogonal')(conv5)      
        bn5 = BatchNormalization()(conv5)
        conv_atv5 = Activation('relu')(bn5)
        drop5 = Dropout(0.5)(conv_atv5)

        up6 = Conv2D(feature_base*8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'Orthogonal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)

        conv6 = Conv2D(feature_base*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'Orthogonal')(merge6)
        conv6 = Conv2D(feature_base*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'Orthogonal')(conv6)

        up7 = Conv2D(feature_base*4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'Orthogonal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)

        conv7 = Conv2D(feature_base*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'Orthogonal')(merge7)
        conv7 = Conv2D(feature_base*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'Orthogonal')(conv7)

        up8 = Conv2D(feature_base*2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'Orthogonal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)

        conv8 = Conv2D(feature_base*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'Orthogonal')(merge8)
        conv8 = Conv2D(feature_base*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'Orthogonal')(conv8)

        up9 = Conv2D(feature_base, 2, activation = 'relu', padding = 'same', kernel_initializer = 'Orthogonal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)

        conv9 = Conv2D(feature_base, 3, activation = 'relu', padding = 'same', kernel_initializer = 'Orthogonal')(merge9)
        conv9 = Conv2D(feature_base, 3, activation = 'relu', padding = 'same', kernel_initializer = 'Orthogonal')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'Orthogonal')(conv9)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

        model = Model(input = inputs, output = conv10)

        model.compile(optimizer = Adam(lr = 1e-5), loss = self.dice_loss, metrics = ['accuracy'])
        print(model.summary())
        return model
    


    def dice_loss(self,y_true, y_pred):
        '''Just another crossentropy'''
        smooth = 1e-2
        l = tf.reduce_sum(y_true*y_true)
        r = tf.reduce_sum(y_pred*y_pred)
        return -(2*tf.reduce_sum(y_true*y_pred)+smooth)/(l+r+smooth)
    
    
    def train(self,start_point=None):
        print("loading data")

        if not os.path.exists(self.working_folder):
            os.mkdir(self.working_folder)            
            
        if not os.path.exists(self.train_folder):
            os.mkdir(self.train_folder)
            
        if not os.path.exists(self.result_folder):
            os.mkdir(self.result_folder)             
        
        self.load_data()
           
        model = None
        if start_point is None:
            model = self.get_unet_2d_bn()            

            print("got new unet")
        else:
            print('loading model from:'+start_point)
            try:
                model = load_model(start_point,custom_objects={'dice_loss': self.dice_loss})
                print("loaded unet")
            except:
                print ('loading failed!')
                return
               
        print("loading data done")
        print('Fitting model...') 
                
        model_checkpoint = ModelCheckpoint(self.result_folder+'/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
        history = model.fit(self.imgs_train, self.imgs_mask_train, batch_size=10, nb_epoch=1000, verbose=1, validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])
        with open(self.result_folder+'/trainHistoryDict', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
            
   
def run():
    myunet = myUnet()
# =============================================================================
#     if want to load previous UNET to train on 
# =============================================================================
    myunet.train('./weights.62--0.87.hdf5')
# =============================================================================
#     if wish to start training on new UNET
# =============================================================================
    myunet.train()

    
    


if __name__ == '__main__':
    run()
