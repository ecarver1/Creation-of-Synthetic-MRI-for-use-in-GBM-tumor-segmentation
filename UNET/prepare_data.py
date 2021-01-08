# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 13:57:44 2018

@author: cliu1
"""

import nibabel as nib
import os
import numpy
import gc

                


def run_folder2(img_data,img_files,rootdir,post_fix):
    for root,subFolders,files in os.walk(rootdir):
        for folder in subFolders:
            
            d=os.path.join(rootdir,folder)
            print(d)
            for root,subFolders,files in os.walk(d):
                for file in files:
                    filename, file_extension = os.path.splitext(file)
                    if file_extension == '.gz':
                        corename, core_extension = os.path.splitext(filename)
                        if corename[-len(post_fix):]==post_fix:
                            img_file = root+'\\'+ file  
                            img = nib.nifti1.load(img_file)
                            data = img.get_fdata()
                            data=trim_and_pad(data, output_size=[256,256,155],dt='float32')
                            if post_fix in filename:
                                img_data.append(data)
                            else:
                                continue
                
    return img_data



def run2(rootdir,outputdir,post_fix):    
    img_data = []
    seg_files = []
    
    for root,subFolders,files in os.walk(rootdir):
        for folder in subFolders:
            print(folder)
            [img_data,seg_files] = run_folder2(img_data,seg_files,os.path.join(root,folder),post_fix)
            
    numpy.save(outputdir + '/'+post_fix+'.npy', numpy.asarray(img_data,dtype='float32'))
    numpy.save(outputdir + '/'+post_fix+'_files.npy', seg_files)

# 1. remove empty slices and reduce the number of slices to 128
# 2. pad the image to 256x256x128
            
def trim_and_pad(img, output_size=[256,256,155],dt='float32'):
            input_img = img
            
             

            
            output_img = numpy.zeros([output_size[0],output_size[1],output_size[2]],dtype=dt)
            #        remove empty slices from the top
#            top = input_img.shape[2]-1
#            for k in range(0,input_img.shape[2]):
#                tk = input_img.shape[2]-k-1
#                if numpy.max(input_img[:,:,tk]) != 0 or tk == output_size[2]-1:
#                    top = tk
#                    break
            
            marx = int((output_size[0]-input_img.shape[0])/2)
            mary = int((output_size[1]-input_img.shape[1])/2)
            marz = int((output_size[2]-input_img.shape[2])/2)
            output_img[marx:marx+input_img.shape[0],mary:mary+input_img.shape[1],marz:marz+input_img.shape[2]] = input_img[:,:,:]
            
          


            return output_img

if __name__== "__main__":
    rootdir = ''
    outputdir = ''
    run2(rootdir,outputdir,"t1")
    gc.collect()
    run2(rootdir,outputdir,"t1ce")
    gc.collect()
    run2(rootdir,outputdir,"t2")
    gc.collect()
    run2(rootdir,outputdir,"flair")
    gc.collect()
    run2(rootdir,outputdir,"seg")
    gc.collect()    
    

    