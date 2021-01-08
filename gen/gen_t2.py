#!/usr/bin/env python

from __future__ import absolute_import, division, print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import utils
import matplotlib.pyplot as plt 
import sys

import scipy.misc
import nibabel as nib
from vtk.util.numpy_support import vtk_to_numpy
from skimage.morphology import closing, square
from scipy import ndimage

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('t2','model','Directory for t2 model')
flags.DEFINE_integer('channel',1,'')
flags.DEFINE_string('out',None,'')
flags.DEFINE_string('name','generation:0','')
flags.DEFINE_float('cth', 0.5, '')

class Model:
    def __init__ (self, path, name='generation:0',prob=False):
        graph = tf.Graph()
        with graph.as_default():
            saver = tf.train.import_meta_graph(path + '.meta')
        if False:
            for op in graph.get_operations():
                for v in op.values():
                    print(v.name)
        inputs = graph.get_tensor_by_name("mask:0")
        outputs = graph.get_tensor_by_name(name)
         
        if prob:
            shape = tf.shape(outputs)
            # softmax
            outputs = tf.reshape(outputs, (-1,2))
            outputs = tf.nn.softmax(outputs)
            outputs = tf.reshape(outputs, shape)
            outputs = tf.slice(outputs, [0,0,0,0,1],[-1,-1,-1,-1,-1])
            outputs = tf.squeeze(outputs,axis=[4])
            pass
        self.prob = prob
        self.path = path
        self.graph = graph
        self.inputs = inputs
        self.outputs = outputs
        self.saver = saver
        self.sess = None
        pass

    def __enter__ (self):
        assert self.sess is None
        self.sess = tf.Session(graph=self.graph)
        self.saver.restore(self.sess, self.path)
        return self

    def __exit__ (self,eType,eValue,eTrace):
        self.sess.close()
        self.sess = None

    def apply (self, noise, batch=1):
        if self.sess is None:
            raise Exception('Model.apply must be run within contest manager')
        return self.sess.run([self.outputs], feed_dict={self.inputs:noise})
    pass

def save (path, images, prob,mask):
    image = images[0,:,:,0]
    prob = prob[0,:,:]
    mask = mask[:,:,0]
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    print(image.shape,mask.shape,prob.shape)
    contours = measure.find_contours(prob, FLAGS.cth)

    prob *= 255
    prob = cv2.cvtColor(prob,cv2.COLOR_GRAY2BGR)
  
    H = max(image.shape[0], prob.shape[0])
    both = np.zeros((H, image.shape[1]+mask.shape[1]+ prob.shape[1],3))
    both[0:mask.shape[0],0:mask.shape[1],:] = mask
    off = mask.shape[1]

    for contour in contours:
        tmp = np.copy(contour[:,0])
        contour[:,0] = contour[:,1]
        contour[:,1] = tmp
        contour = contour.reshape((1,-1,2)).astype(np.int32)
        cv2.polylines(image,contour, True, (0,255,0))
        cv2.polylines(prob, contour, True, (0,255,0))

    both[0:image.shape[0],off:(off+image.shape[1]),:] = image
    off += image.shape[1]
    both[0:prob.shape[0], off:(off+prob.shape[1]),:] = prob
    cv2.imshow('image',both)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('1.jpg',both)

def myaffine (rows,cols,matrix):
    rows = np.expand_dims(rows,axis=0)
    cols = np.expand_dims(cols,axis=0)
    cor = np.append(rows,cols,axis=0)
    cor = np.append(cor,np.ones([1,rows.shape[1]]),axis=0)
    cor = np.matmul(matrix,cor)
    rows = (cor[0,:]).astype(np.int8)
    cols = (cor[1,:]).astype(np.int8)
    return rows,cols

def main(_):
    path = r'./HGG'
    try:os.makedirs('./generation')
    except:pass
    try:os.makedirs('./generation/t2')
    except:pass
    try:os.makedirs('./generation/label')
    except:pass
    

    with Model(FLAGS.t2, name=FLAGS.name,prob=False) as modelt2:
        for gz in os.listdir(path):
            if not '_seg.nii.gz' in gz:continue
            masks = nib.load(os.path.join(path,gz))
            masks = masks.get_fdata()
            t2name = gz.split('_s')[0]+'_t2.nii.gz'
            t2s = nib.load(os.path.join(path,t2name))
            t2s = t2s.get_fdata()
        
            # define random mask params
            theta = np.random.randint(-80,80)
            dx = np.random.randint(-10,10)
            dy = np.random.randint(-10,10)
            dz = np.random.randint(-5,5)
            fliplr = np.random.randint(0,2)
            flipud = np.random.randint(0,2)
            scale = np.random.uniform(0.7,1.3)
        
            print(fliplr,flipud,dx,dy,theta)
        
            name =  gz.split('_s')[0]

            nibt2 = np.zeros([256,256,155])
            niblabel = np.zeros([256,256,155])

            for i in range(0,155):
                if t2s[:,:,i].any() == 0:continue # no value for images, contine
                im = np.zeros([256,256]) # padding to 256
                im[8:248,8:248] = t2s[:,:,i]
            
                mask = masks[:,:,i] 
                labels = np.zeros([256,256,3])
                mask_ = np.zeros([256,256])
                mask_[8:248,8:248] = mask
                mask = mask_ # pad image mask to 256 256
            
                mask_ = ndimage.shift(mask,[dx,dy]).astype(np.int8)
                mask_ = ndimage.rotate(mask,theta,order=0,reshape=False).astype(np.int8)
                if fliplr:mask_ = np.fliplr(mask_)
                if flipud:mask_ = np.flipud(mask_) 
                
                #continue
                #first three channels for mask
                rows,cols = np.where(mask_ == 1)
                labels[rows,cols,0] = 1
                rows,cols = np.where(mask_ == 2)
                labels[rows,cols,1] = 1
                rows,cols = np.where(mask_ == 4)
                labels[rows,cols,2] = 1
                
                # generate mask for normal part - based on threshold
                th = np.amax(im)/6
                th1 = th*4.5
                th2 = th*3
                th3 = th*1.5
                
                total = closing(im>0,square(3)).astype(np.int16)
                trows,tcols = np.where(total == 0)
                p1 = closing(im>th1,square(3)).astype(np.int16) # >300
                p2 = closing(im>th2,square(3)).astype(np.int16)-p1 # 200-300
                p3 = closing(im>th3,square(2)).astype(np.int16)-p1-p2 # 100-200
                p4 = total-p1-p2-p3 #0-100
                
                nrows,ncols = np.where(mask_) #new lesion
                orows,ocols = np.where(mask) #old lesion
                
                # old lesion set to nomal part (assuem p3), new lesion set to zero
                p1[orows,ocols] = 0
                p2[orows,ocols] = 0
                p3[orows,ocols] = 1
                p4[orows,ocols] = 0

                p1[nrows,ncols] = 0
                p2[nrows,ncols] = 0
                p3[nrows,ncols] = 0
                p4[nrows,ncols] = 0

                labels[trows,tcols,:] = 0 # lesion out of brain set to zero
             
                labels = np.append(labels,np.expand_dims(p1,axis=2),axis=2)
                labels = np.append(labels,np.expand_dims(p2,axis=2),axis=2)
                labels = np.append(labels,np.expand_dims(p3,axis=2),axis=2)
                labels = np.append(labels,np.expand_dims(p4,axis=2),axis=2)
                                
                in_label = np.expand_dims(labels,axis=0)
                in_label = np.concatenate((in_label,np.expand_dims(1-np.sum(in_label,axis=3),axis=3)),axis=3)
            
                GG = modelt2.apply(in_label)
                output = np.array(GG)
                output = output[0,:,:,:,:]
                nibt2[:,:,i] = np.multiply(output[0,:,:,0],
                    labels[:,:,0]+labels[:,:,1]+labels[:,:,2]+labels[:,:,3]+labels[:,:,4]+labels[:,:,5]+labels[:,:,6])

                niblabel[:,:,i] = labels[:,:,0]+2*labels[:,:,1]+4*labels[:,:,2]

            # save nib file
            niblabel_path = './generation/label/'+name+'.nii.gz'
            nibt2_path = niblabel_path.replace('label','t2')
            
            nib.save(nib.Nifti1Image(nibt2, np.eye(4)),nibt2_path)
            nib.save(nib.Nifti1Image(niblabel, np.eye(4)),niblabel_path)
        #break
tf.app.run()
