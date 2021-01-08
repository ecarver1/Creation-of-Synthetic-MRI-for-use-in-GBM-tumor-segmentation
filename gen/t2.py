#Source codes from Qifeng Chen and Vladlen Koltun. Photographic Image Synthesis with Cascaded Refinement Networks. In ICCV 2017.

from __future__ import division
import os,time,scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
from gan_nets import mae_criterion as criterionGAN
from gan_nets import dis

def lrelu(x):
    return tf.maximum(0.2*x,x)

def build_net(ntype,nin,nwb=None,name=None):
    if ntype=='conv':
        return tf.nn.relu(tf.nn.conv2d(nin,nwb[0],strides=[1,1,1,1],padding='SAME',name=name)+nwb[1])
    elif ntype=='pool':
        return tf.nn.avg_pool(nin,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def get_weight_bias(vgg_layers,i):
    weights=vgg_layers[i][0][0][2][0][0]
    weights=tf.constant(weights)
    bias=vgg_layers[i][0][0][2][0][1]
    bias=tf.constant(np.reshape(bias,(bias.size)))
    return weights,bias

def build_vgg19(input,reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    net={}
    vgg_rawnet=scipy.io.loadmat('./imagenet-vgg-verydeep-19.mat')
    vgg_layers=vgg_rawnet['layers'][0]
    #print('input:',input.shape)
    net['input']=input-np.array([0.0, 0.0, 0.0]).reshape((1,1,1,3)) 
    #print('net input',net['input'].shape)
    
    net['conv1_1']=build_net('conv',net['input'],get_weight_bias(vgg_layers,0),name='vgg_conv1_1')
    net['conv1_2']=build_net('conv',net['conv1_1'],get_weight_bias(vgg_layers,2),name='vgg_conv1_2')
    net['pool1']=build_net('pool',net['conv1_2'])
    net['conv2_1']=build_net('conv',net['pool1'],get_weight_bias(vgg_layers,5),name='vgg_conv2_1')
    net['conv2_2']=build_net('conv',net['conv2_1'],get_weight_bias(vgg_layers,7),name='vgg_conv2_2')
    net['pool2']=build_net('pool',net['conv2_2'])
    net['conv3_1']=build_net('conv',net['pool2'],get_weight_bias(vgg_layers,10),name='vgg_conv3_1')
    net['conv3_2']=build_net('conv',net['conv3_1'],get_weight_bias(vgg_layers,12),name='vgg_conv3_2')
    net['conv3_3']=build_net('conv',net['conv3_2'],get_weight_bias(vgg_layers,14),name='vgg_conv3_3')
    net['conv3_4']=build_net('conv',net['conv3_3'],get_weight_bias(vgg_layers,16),name='vgg_conv3_4')
    net['pool3']=build_net('pool',net['conv3_4'])
    net['conv4_1']=build_net('conv',net['pool3'],get_weight_bias(vgg_layers,19),name='vgg_conv4_1')
    net['conv4_2']=build_net('conv',net['conv4_1'],get_weight_bias(vgg_layers,21),name='vgg_conv4_2')
    net['conv4_3']=build_net('conv',net['conv4_2'],get_weight_bias(vgg_layers,23),name='vgg_conv4_3')
    net['conv4_4']=build_net('conv',net['conv4_3'],get_weight_bias(vgg_layers,25),name='vgg_conv4_4')
    net['pool4']=build_net('pool',net['conv4_4'])
    net['conv5_1']=build_net('conv',net['pool4'],get_weight_bias(vgg_layers,28),name='vgg_conv5_1')
    net['conv5_2']=build_net('conv',net['conv5_1'],get_weight_bias(vgg_layers,30),name='vgg_conv5_2')
    net['conv5_3']=build_net('conv',net['conv5_2'],get_weight_bias(vgg_layers,32),name='vgg_conv5_3')
    net['conv5_4']=build_net('conv',net['conv5_3'],get_weight_bias(vgg_layers,34),name='vgg_conv5_4')
    net['pool5']=build_net('pool',net['conv5_4'])
    return net

def recursive_generator(label,sp):
    dim=512 if sp>=128 else 1024
    if sp==4:
        input=label
    else:
        downsampled=tf.image.resize_area(label,(sp//2,sp//2),align_corners=False)
        input=tf.concat([tf.image.resize_bilinear(recursive_generator(downsampled,sp//2),(sp,sp),align_corners=True),label],3)
    net=slim.conv2d(input,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv1')
    net=slim.conv2d(net,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv2')
    net=tf.add(net,slim.conv2d(input,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,scope='g_'+str(sp)+'_side'),
    name='g_'+str(sp)+'res')
    if sp==256:
        net=slim.conv2d(net,9,[1,1],rate=1,activation_fn=None,scope='g_'+str(sp)+'_conv100') # one channel
        net = tf.transpose(net,perm=[3,1,2,0]) # one channel
    return net

def compute_error(real,fake,label):
    return tf.reduce_mean(label*tf.expand_dims(tf.reduce_mean(tf.abs(fake-real),reduction_indices=[3]),-1),reduction_indices=[1,2])#diversity loss

'''
def dis (G,reuse,alpha,name):
    with tf.variable_scope(name,reuse=reuse):

        net = slim.conv2d(G,64,5,2,padding='same',activation_fn=None)
        net = tf.maximum(alpha*net,net)
            
        net = slim.batch_norm(slim.conv2d(net,128,5,2,padding='same',activation_fn=None)) #16
        net = tf.maximum(alpha*net,net)
        net = slim.batch_norm(slim.conv2d(net,256,5,2,padding='same',activation_fn=None)) #8
        net = tf.maximum(alpha*net,net)
        net = slim.batch_norm(slim.conv2d(net,512,5,2,padding='same',activation_fn=None)) #4
        net = tf.maximum(alpha*net,net)
            
        net = slim.conv2d(net, 1, 3, 1, activation_fn =None) #1
        #net = tf.reduce_mean(net,axis = [1,2],keepdims = True)
        #net = tf.squeeze(net, [1,2]) 
            
        output = tf.nn.sigmoid(net)
        return net,tf.identity(output,'output')
'''
sess=tf.Session()
is_training=True
sp=256 #spatial resolution: 256x256
with tf.variable_scope(tf.get_variable_scope()):
    label=tf.placeholder(tf.float32,[1,256,256,8],name = 'mask')  # input
    real_image=tf.placeholder(tf.float32,[1,256,256,1],name = 'image')  # ground truth
    print('building generator...')
    generator=recursive_generator(label,sp)
    generator = tf.identity(generator,'generation')
    
    dis_real,_ = dis(real_image,reuse = False,alpha = 0.2,name="discriminator")
    dis_fake,_ = dis(generator,reuse = True,alpha = 0.2,name="discriminator")

    print('building vgg_real...')
    vgg_real=build_vgg19(real_image,reuse=False)
    print('building vgg_fake...')
    vgg_fake=build_vgg19(generator,reuse=True)
    
    p0=compute_error(vgg_real['input'],vgg_fake['input'],label)
    p1=compute_error(vgg_real['conv1_2'],vgg_fake['conv1_2'],label)*0.6
    p2=compute_error(vgg_real['conv2_2'],vgg_fake['conv2_2'],tf.image.resize_area(label,(sp//2,sp//2)))*0.5
    p3=compute_error(vgg_real['conv3_2'],vgg_fake['conv3_2'],tf.image.resize_area(label,(sp//4,sp//4)))*0.5
    p4=compute_error(vgg_real['conv4_2'],vgg_fake['conv4_2'],tf.image.resize_area(label,(sp//8,sp//8)))*0.3
    p5=compute_error(vgg_real['conv5_2'],vgg_fake['conv5_2'],tf.image.resize_area(label,(sp//16,sp//16)))/0.15
    

    #weights lambda are collected at 100th epoch
    content_loss=p0+p1+p2+p3+p4+p5
    gan_loss = criterionGAN(dis_fake, tf.ones_like(dis_fake))*np.random.uniform(0.9,1)
    #adjust coefficient
    G_loss=tf.reduce_sum(tf.reduce_min(content_loss,reduction_indices=0))*0.999+tf.reduce_sum(tf.reduce_mean(content_loss,reduction_indices=0))*0.001
    #+0.1*tf.reduce_sum(tf.reduce_min(gan_loss,reduction_indices=0))*0.999
    #+0.1*tf.reduce_sum(tf.reduce_mean(gan_loss,reduction_indices=0))*0.001
    
    d_loss_real = criterionGAN(dis_real, tf.ones_like(dis_real))*np.random.uniform(0.9,1)
    d_loss_fake = criterionGAN(dis_fake, tf.zeros_like(dis_fake))*np.random.uniform(0.9,1) 
    D_loss = tf.reduce_sum(tf.reduce_min(d_loss_real,reduction_indices=0))*0.999+tf.reduce_sum(tf.reduce_mean(d_loss_real,reduction_indices=0))*0.001+tf.reduce_sum(tf.reduce_min(d_loss_fake,reduction_indices=0))*0.999+tf.reduce_sum(tf.reduce_mean(d_loss_fake,reduction_indices=0))*0.001

lr=tf.placeholder(tf.float32)
G_opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss,var_list=[var for var in tf.trainable_variables() if var.name.startswith('g_')])
D_opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(D_loss,var_list=[var for var in tf.trainable_variables() if var.name.startswith('dis')])
saver=tf.train.Saver(max_to_keep=5)
sess.run(tf.global_variables_initializer())

ckpt=tf.train.get_checkpoint_state("result_t2")
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)

## load data and label
labelpath='./label'
labels = glob(os.path.join(labelpath,'*.npy'))

# create result folder
if os.path.exists('./result_t2'): pass
else: os.mkdir('./result_t2')

if is_training:
    for epoch in range(1,101):
        if os.path.isdir("result_t2/%04d"%epoch):
            continue
        cnt=0
        np.random.shuffle(labels)
        for labelname in labels:
            st=time.time()
            cnt+=1
            label_images = np.load(labelname)
            label_images = np.expand_dims(label_images,axis=0)
            
            data = labelname.replace('label','realt2') # data path
            input_images = np.load(data)
            input_images = np.expand_dims(input_images,axis=0)
            input_images = np.expand_dims(input_images,axis=3)
           
            _ = sess.run([D_opt],feed_dict={label:np.concatenate((label_images,np.expand_dims(1-np.sum(label_images,axis=3),axis=3)),axis=3),real_image:input_images,lr:1e-4})
            _,G_current,l0,l1,l2,l3,l4,l5=sess.run([G_opt,G_loss,p0,p1,p2,p3,p4,p5],
                    feed_dict={label:np.concatenate((label_images,np.expand_dims(1-np.sum(label_images,axis=3),axis=3)),axis=3),
                            real_image:input_images,lr:1e-4})
            print("%d %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f"%(epoch,cnt,G_current,np.mean(l0),np.mean(l1),
                np.mean(l2),np.mean(l3),np.mean(l4),np.mean(l5),time.time()-st))
        
        os.makedirs("result_t2/%04d"%epoch)
        saver.save(sess,"result_t2/model.ckpt")
        
        # sample during training
        GG = sess.run([generator], feed_dict={label:np.concatenate((label_images,
            np.expand_dims(1-np.sum(label_images,axis=3),axis=3)),axis=3)})
        
        #output=np.minimum(np.maximum(GG,0.0),255.0)
        output = np.array(GG)
        output = output[0,:,:,:,:]
        upper = np.concatenate((output[0,:,:,:],output[1,:,:,:],output[2,:,:,:]),axis=1)
        middle=np.concatenate((output[3,:,:,:],output[4,:,:,:],output[5,:,:,:]),axis=1)
        bottom=np.concatenate((output[6,:,:,:],output[7,:,:,:],output[8,:,:,:]),axis=1)
        im = np.concatenate((upper,middle,bottom),axis=0)
        savepath = "result_t2/%04d/%s_output.jpg"%(epoch,labelname.split('\\')[-1].split('.')[0])
        scipy.misc.imsave(savepath,im[:,:,0])
        
        # also save model per 10 epoches
        if epoch%10==0:
            saver.save(sess,"result_t2/%04d/model.ckpt"%epoch)
