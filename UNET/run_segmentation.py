# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 09:10:25 2018

@author: cliu1
"""
import keras
import medpy
from medpy import metric
from medpy import io
from medpy import filter
import numpy as np
import tensorflow as tf
import os
import nibabel as nib
import shutil
import glob

#import matplotlib.pyplot as plt
#import matplotlib.colors as colors

def dice_loss(y_true, y_pred):
    '''Just another crossentropy'''
    smooth = 1e-2
    l = tf.reduce_sum(y_true*y_true)
    r = tf.reduce_sum(y_pred*y_pred)
    return -(2*tf.reduce_sum(y_true*y_pred)+smooth)/(l+r+smooth)

def normalize_image(im):
    avg = np.mean(im,dtype='float32')
    std = np.std(im,dtype='float32')
    return (im-avg)/std
 
def load_data(img_file,normalize = True):
    ori_imgs = np.load(img_file)

    imgs = np.swapaxes(np.swapaxes(ori_imgs,1,3),2,3)
        
    img_rows = imgs.shape[2]
    img_cols = imgs.shape[3]
    
    train_count = imgs.shape[0]*imgs.shape[1]
    imgs=np.reshape(imgs,[train_count,img_rows,img_cols,1])
    
    if normalize:
        for i in range(0,train_count):
            imgs[i,:,:,0] = normalize_image(imgs[i,:,:,0])
    
    return ori_imgs,imgs

def predict_evaluate_single_img(img_files,model,label,output_file=None):
    
    img_test = load_image(img_files[0])
    for i in range(1,len(img_files)):
        img_test = np.append(img_test,load_image(img_files[i]),axis=3)
        
    img_predict = model.predict(img_test.astype(np.float16), batch_size=1, verbose=1)
    predict = np.swapaxes(np.swapaxes(np.squeeze(img_predict,axis=3),0,2),0,1)
    th = 0.5
    predict[predict>=th] = 1
    predict[predict<th] = 0
    predict = medpy.filter.binary.largest_connected_component(predict[4:244,4:244,:])
    dc = medpy.metric.binary.dc(predict,label)
    sens = medpy.metric.binary.sensitivity(predict,label)
    spec = medpy.metric.binary.specificity(predict,label)
    hd = medpy.metric.binary.hd(predict,label)
    
    if output_file is not None:
        with open(output_file, "a") as myfile:
            myfile.write(img_files[0]+','+str(dc)+','+str(sens)+','+str(spec)+','+str(hd)+'\n')
    
    return dc,sens,spec,hd
    
def load_image(img_file):
    img,h = medpy.io.load(img_file)
    
    swap_img = np.swapaxes(np.swapaxes(img,0,2),1,2)
    swap_img = normalize_image(swap_img.astype(float))
    expand_img = np.zeros([swap_img.shape[0],256,256])
    expand_img[:,4:244,4:244] = swap_img
    return np.expand_dims(expand_img,axis=3).astype(np.float16)

def get_prediction(img_files,model):    
    img_test = load_image(img_files[0])
    
    for i in range(1,len(img_files)):
        img_test = np.append(img_test,load_image(img_files[i]),axis=3)
        
    img_predict = model.predict(img_test.astype(np.float16), batch_size=1, verbose=1)
    predict = np.swapaxes(np.swapaxes(np.squeeze(img_predict,axis=3),0,2),0,1)
    th = 0.5
    predict[predict>=th] = 1
    predict[predict<th] = 0
    if 1 in predict:
        predict = medpy.filter.binary.largest_connected_component(predict[4:244,4:244,:])
    else:
        predict = predict[4:244,4:244,:]
    
    return predict
    
# use all modalities provided for one prediction
def predict_single_img(img_files,model,output_file):
    
    predict = get_prediction(img_files,model)
    
#    imgs_show_pair(img,label,predict)
    
    if output_file[-3:] == 'npy':
        np.save(output_file,predict.astype(int))
    elif output_file[-3:] == '.gz' or output_file[-3:] == 'nii':
        nib.nifti1.save(nib.Nifti1Image(predict.astype(int),affine=np.eye(4)),output_file)

# use all modalities provided for one prediction
# the 1st channel was the difference between the 1st and 2nd modality
def predict_single_img2(img_files,model,output_file):
        
    img_test1 = load_image(img_files[0])
    img_test2 = load_image(img_files[1])
    img_test = img_test2 - img_test1
    img_test = np.append(img_test,img_test2,axis=3)
    
    for i in range(2,len(img_files)):
        img_test = np.append(img_test,load_image(img_files[i]),axis=3)
        
    img_predict = model.predict(img_test.astype(np.float16), batch_size=1, verbose=1)
    predict = np.swapaxes(np.swapaxes(np.squeeze(img_predict,axis=3),0,2),0,1)
    th = 0.5
    predict[predict>=th] = 1
    predict[predict<th] = 0
    if 1 in predict:
        predict = medpy.filter.binary.largest_connected_component(predict[4:244,4:244,:])
    else:
        predict = predict[4:244,4:244,:]
    
#    imgs_show_pair(img,label,predict)
    
    nib.nifti1.save(nib.Nifti1Image(predict.astype(int),affine=np.eye(4)),output_file)

def run_folder(parent,folder,model,predict_type,output_folder):
    rootdir = os.path.join(parent,folder)
    t1_file = os.path.join(rootdir,folder+'_t1.nii.gz')
    t1ce_file = os.path.join(rootdir,folder+'_t1ce.nii.gz')
    t2_file = os.path.join(rootdir,folder+'_t2.nii.gz')
    flair_file = os.path.join(rootdir,folder+'_flair.nii.gz')
    seg_file = os.path.join(rootdir,folder+'_seg.nii.gz')

    result_img_file = os.path.join(output_folder,folder+'.nii.gz')
    
    if os.path.exists(seg_file):
        label,hl = medpy.io.load(seg_file)
        if predict_type == 'wt':
            label[label>0] = 1
        elif predict_type == 'et':
            label[label==2] = 0
            label[label==1] = 0
            label[label==4] = 1
        elif predict_type == 'tc':
            label[label==2] = 0
            label[label==4] = 1
            

                                    
    img_files = [t1_file,t1ce_file,t2_file,flair_file]
    
    if not os.path.exists(result_img_file):
        print(predict_single_img(img_files,model,result_img_file))

    
# generate seg image by combining results stored in the original folder structure
def generate_mask(parent,folder,output_folder):    
    rootdir = os.path.join(parent,folder)
    ref_file = os.path.join(rootdir,folder+'_flair.nii.gz')
    wt_file = os.path.join(output_folder,folder+'_wt.nii.gz')
    tc_file = os.path.join(output_folder,folder+'_tc.nii.gz')
    et_file = os.path.join(output_folder,folder+'_et.nii.gz')
    upload_folder = os.path.join(output_folder,'upload')
    
    if not os.path.exists(upload_folder):
        os.mkdir(upload_folder)
        
    result_file = os.path.join(upload_folder,folder+'.nii.gz')
    
    if not os.path.exists(result_file):    
        wt = nib.nifti1.load(wt_file).get_fdata()
        tc = nib.nifti1.load(tc_file).get_fdata()
        et = nib.nifti1.load(et_file).get_fdata()
        
        if np.array_equal(wt.shape,tc.shape) and np.array_equal(wt.shape,et.shape):      
            result = wt*2

            tc_wt = np.logical_and(tc>0, wt>0)
            result[tc_wt] = 1
            result[np.logical_and(et>0, tc_wt)] = 4
            
            result = result.astype(np.int16)
            
            ref = nib.nifti1.load(ref_file)
            ref.header['data_type'] = 'int16'
            nib.nifti1.save(nib.Nifti1Image(result,ref.affine, ref.header),result_file)
        else:
            print('array not of the same shape:'+folder)

def combine_masks(wt,tc,et):
    # only consider wt masks at certain slices
    wt[:,:,0:20] = 0
    wt = medpy.filter.binary.largest_connected_component(wt)
    
    if np.array_equal(wt.shape,tc.shape) and np.array_equal(wt.shape,et.shape):      
        result = wt*2

        tc_wt = np.logical_and(tc>0, wt>0)
        if True in tc_wt:
            result[tc_wt] = 1
        else:
            result = wt
            
        et_tc_wt = np.logical_and(et>0, tc_wt)
        if True in et_tc_wt:
            result[et_tc_wt] = 4
        else:
            result = wt*4
        
        return result.astype(np.int16)
    else:
        print('array not of the same shape:'+folder)
        return None
    
# generate seg image by combining results stored in the copied folder structure (no sub folders)
def generate_mask2(rootdir,folder,output_folder):  
    wt_file = rootdir+'/wt/'+folder+'.nii.gz'
    tc_file = rootdir+'/tc/'+folder+'.nii.gz'
    et_file = rootdir+'/et/'+folder+'.nii.gz'
    ref_file = wt_file
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
        
    result_file = os.path.join(output_folder,folder+'.nii.gz')
    
    if (not os.path.exists(result_file)) and os.path.exists(wt_file) and os.path.exists(tc_file) and os.path.exists(et_file):    
        wt = nib.nifti1.load(wt_file).get_fdata()
        tc = nib.nifti1.load(tc_file).get_fdata()
        et = nib.nifti1.load(et_file).get_fdata()       
            
        result = combine_masks(wt,tc,et)
            
        if result is not None:
            ref = nib.nifti1.load(ref_file)
            ref.header['data_type'] = 'int16'
            nib.nifti1.save(nib.Nifti1Image(result,ref.affine, ref.header),result_file)
            

def run_root_folder(rootdir,model_file,predict_type,output_root):
    
    model = keras.models.load_model(model_file,custom_objects={'dice_loss': dice_loss})
    
    output_folder = os.path.join(output_root,predict_type)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
        
    for root,subFolders,files in os.walk(rootdir):
        for folder in subFolders:
            run_folder(rootdir,folder,model,predict_type,output_folder)

    
def run_single(wt_model_file,et_model_file,tc_model_file,t1_file,t2ce_file,t2_file,flair_file,result_file):
    wt_model = keras.models.load_model(wt_model_file,custom_objects={'dice_loss': dice_loss})
    wt = get_prediction([t1_file,t1ce_file,t2_file,flair_file],wt_model)
    et_model = keras.models.load_model(et_model_file,custom_objects={'dice_loss': dice_loss})
    et = get_prediction([t1_file,t1ce_file,t2_file,flair_file],et_model)
    tc_model = keras.models.load_model(tc_model_file,custom_objects={'dice_loss': dice_loss})
    tc = get_prediction([t1_file,t1ce_file,t2_file,flair_file],tc_model)
    
    ref_file = t1_file
    result = combine_masks(wt,tc,et)
        
    if result is not None:
        ref = nib.nifti1.load(ref_file)
        ref.header['data_type'] = 'int16'
        nib.nifti1.save(nib.Nifti1Image(result,ref.affine, ref.header),result_file)
    
    
if __name__== "__main__":
    
    et_model_file = './brats2018/et.hdf5'
    wt_model_file = './brats2018/wt.hdf5'
    tc_model_file = './brats2018/tc.hdf5'
    t1_file = './data/t1.nii.gz'
    t2_file = './data/t2.nii.gz'
    t1ce_file = './data/t1ce.nii.gz'
    flair_file = './data/flair.nii.gz'
    result_file = './data/results/tumor_hfhs_brats2018_class.nii.gz'
    
    run_single(wt_model_file,et_model_file,tc_model_file,t1_file,t1ce_file,t2_file,flair_file,result_file)

    