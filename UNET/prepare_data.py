# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 13:57:44 2018

@author: ecarver1
"""

import nibabel as nib
import os
import numpy
import gc


    




def run3(rootdir,outputdir,post_fix,outputdir_nifti):    
    img_data = []
    print(post_fix)
    img_data=run_folder_no_folder_folder(img_data,img_files,rootdir,post_fix,outputdir_nifti)

    numpy.save(outputdir + '/'+post_fix+'.npy', numpy.asarray(img_data,dtype='float32'))


# 1. remove empty slices and reduce the number of slices to 128
# 2. pad the image to 256x256x128
def trim_and_pad(data, file_path,post_fix, output_size=[256,256,64],dt='uint16'):

            input_img = data

            seg_file = file_path.replace(post_fix+'.nii.gz', 'seg.nii.gz')
            print(seg_file)
            seg_img = nib.nifti1.load(seg_file)
            input_imgg = seg_img.get_fdata()
            
            output_img = numpy.zeros([output_size[0],output_size[1],output_size[2]],dtype=dt)
            #        remove empty slices from the top
            top = input_imgg.shape[2]-1
            for k in range(0,input_imgg.shape[2]):
                tk = input_imgg.shape[2]-k-1
                if numpy.max(input_imgg[:,:,tk]) != 0 or tk == output_size[2]-1:
                    top = tk
                    break
            print(top)
            marx = int((output_size[0]-input_img.shape[0])/2)
            mary = int((output_size[1]-input_img.shape[1])/2)
            output_img[marx:marx+input_img.shape[0],mary:mary+input_img.shape[1],:] = input_img[:,:,top-output_size[2]+1:top+1]
            print(output_img.shape,numpy.mean(output_img))
            return output_img,top



def run_folder_no_folder_folder(img_data,img_files,rootdir,post_fix,outputdir_nifti):

    for root,subFolders,files in os.walk(rootdir):
          for file in files:
              g=os.path.join(rootdir,file)

              filename, file_extension = os.path.splitext(file)
              if file_extension == '.gz':
                  corename, core_extension = os.path.splitext(filename)
                  if corename[-len(post_fix):]==post_fix:
                      if 'N4' not in filename:
                            if post_fix=='seg':
                                
                                img_file = g
                                print(img_file)
                                img = nib.nifti1.load(img_file)
                                data = img.get_fdata()
                                data,top=trim_and_pad(data, g,post_fix, output_size=[256,256,64],dt='float32')


                            else:
                                img_file = g
                                print(img_file)
                                img = nib.nifti1.load(img_file)
                                data = img.get_fdata()

                                data,top=trim_and_pad(data, g,post_fix, output_size=[256,256,64],dt='float32')
                                print(data.shape)
                                data=numpy.asarray(data)
                            if post_fix in filename:
                                img_data.append(data)
                        
                            else:
                                continue
                
    return img_data
if __name__== "__main__":
    rootdir = '/Brats18_train'
    outputdir ='/Brats18_train_output'
    outputdir_nifti='/Brats18_train_output'
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    if not os.path.exists(outputdir_nifti):
        os.mkdir(outputdir_nifti)
    img_data=[]
    img_files=[]

    run3(rootdir,outputdir,"t1",outputdir_nifti)
    gc.collect()
    run3(rootdir,outputdir,"t1ce",outputdir_nifti)
    gc.collect()
    run3(rootdir,outputdir,"t2",outputdir_nifti)
    gc.collect()
    run3(rootdir,outputdir,"flair",outputdir_nifti)
    gc.collect()
    run3(rootdir,outputdir,"seg",outputdir_nifti)
    gc.collect()    
    

    

    
    
