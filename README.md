# Creation-of-Synthetic-MRI-for-use-in-GBM-tumor-segmentation

Codes adapted from: Qifeng Chen and Vladlen Koltun. Photographic Image Synthesis with Cascaded Refinement Networks. In ICCV 2017.

Used for the generation of differenct synthetic MRI modalities with manipulation. Given T2 modality as an example.

Set up
- Tensorflow 1.7
-CUDA 9.0
-cudnn 9.0

GAN:
Training 
Use t2.py. This is for the train of T2 modality, change the files accordingly in 'real_t2' to train other modalities.

Testing 
python gen_t2.py --t2 PAHT_TO_MODEL. The generation of T2 modality. The code will generate synthetic T2s with
lesions manipulated (fliplr, flipup, translation and rotation). Outputs are new lesion contours and synthetic T2s in NIfTI format.

UNET:

Prepare Data: Converts training nifti files into cropped numpy array
UNET: Trains segmentation UNET 
Run Segmentation: Generates Whole Tumor/Enhancing Tumor/Tumor Core contours 

Citation
If you use our code for research, please cite our paper: "Automatic brain tumor segmentation and overall survival prediction using machine learning algorithms"
