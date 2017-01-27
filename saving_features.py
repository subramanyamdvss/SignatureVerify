


import tensorflow as tf
import cv2
slim = tf.contrib.slim
from PIL import Image
from create_and_fill import *
from inception_resnet_v2 import *
from inception_preprocessing import *
import numpy as np
from os import listdir
from os.path import isfile, join
# %matplotlib inline
# import matplotlib.pyplot as plt
# import scipy.misc
# path to your inception checkpoint file
checkpoint_file = '/home/surya/Documents/inception_resnet_v2_2016_08_30.ckpt'
# path to your dataset
dataset_dir="/home/surya/Documents/signature"
# path to your new dataset for features.
dataset_dir_new="/home/surya/Documents/signature_features"

#number of ids
n_id=38
#call to create new dataset
# create_fdirs("/home/surya/Documents/signature_features")




def creating_features(sess,logits,end_points,tmp,input_tensor):
    

    prelog=[]
    for i,im in enumerate(tmp):
        im = [im]
        im = np.array(im)
        print type(im)
        prelogits,logits_aft= sess.run([end_points['PreLogitsFlatten'],logits],feed_dict={input_tensor:im})
        prelog.append(prelogits)
    return prelog

        

def merge_images(n_id,dataset_dir,dataset_dir_new):
    
    numf=31637
    numg=1

    # Load the model
    sess = tf.Session()
    arg_scope = inception_resnet_v2_arg_scope()
    with slim.arg_scope(arg_scope):
        input_tensor=tf.placeholder(tf.float32, (1,299,299,3), name=None)
        logits, end_points = inception_resnet_v2(input_tensor, is_training=False)
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_file)
    
    # used to produce forged comparisions
    for i in range(27,n_id+1):

        for j,f1 in enumerate(listdir(os.path.join(dataset_dir,"id_%d/genuine" %(i)))):

            for k,f2 in enumerate(listdir(os.path.join(dataset_dir,"id_%d/forge" %(i)))):
                
                f1p=os.path.join(dataset_dir,"id_%d/genuine" %(i))
                f1p=os.path.join(f1p,f1)
                f2p=os.path.join(dataset_dir,"id_%d/forge" %(i))
                f2p=os.path.join(f2p,f2)
                im1 = cv2.imread(f1p)
                im2 = cv2.imread(f2p)
                im1gr = cv2.resize(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY),(299,299))
                im2gr = cv2.resize(cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY),(299,299))
                tmp1=np.dstack((im1gr,im2gr,np.zeros((299,299))))
                tmp1 = 2*(tmp1/255.0)-1.0
                
                # images which are flipped left  right , up  down 
                tmp1 = np.dstack((im1gr,im2gr,np.zeros((299,299))))
                tmp2 = np.flipud(tmp1)
                tmp3 = np.fliplr(tmp1)
                tmp4 = np.fliplr(tmp2)
                tmp=[tmp1,tmp2,tmp3,tmp4]
                prelogitsf = creating_features(sess,logits,end_points,tmp,input_tensor)
                crntf = join(dataset_dir_new,"forge")
                np.save(join(crntf,"f%d.npy") %(numf),prelogitsf[0])
                numf+=1
                np.save(join(crntf,"f%d.npy") %(numf),prelogitsf[1])
                numf+=1
                np.save(join(crntf,"f%d.npy") %(numf),prelogitsf[2])
                numf+=1
                np.save(join(crntf,"f%d.npy") %(numf),prelogitsf[3])
                numf+=1

    # used to produce genuine comparisions
    # for i in range(1,n_id+1):

    #     for j,f1 in enumerate(listdir(os.path.join(dataset_dir,"id_%d/genuine" %(i)))):

    #         for k,f2 in enumerate(listdir(os.path.join(dataset_dir,"id_%d/genuine" %(i)))):

                
    #             f1p=os.path.join(dataset_dir,"id_%d/genuine" %(i))
    #             f1p=os.path.join(f1p,f1)
    #             f2p=os.path.join(dataset_dir,"id_%d/genuine" %(i))
    #             f2p=os.path.join(f2p,f2)
    #             im1 = cv2.imread(f1p)
    #             im2 = cv2.imread(f2p)
    #             im1gr = cv2.resize(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY),(299,299))
    #             im2gr = cv2.resize(cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY),(299,299))
    #             tmp1=np.dstack((im1gr,im2gr,np.zeros((299,299))))
    #             tmp1 = 2*(tmp1/255.0)-1.0
                
    #             # images which are flipped left  right , up  down 
    #             tmp1 = np.dstack((im1gr,im2gr,np.zeros((299,299))))
    #             tmp2 = np.flipud(tmp1)
    #             tmp3 = np.fliplr(tmp1)
    #             tmp4 = np.fliplr(tmp2)
    #             tmp=[tmp1,tmp2,tmp3,tmp4]
    #             prelogitsg = creating_features(sess,logits,end_points,tmp,input_tensor)
    #             crntg = join(dataset_dir_new,"genuine")
    #             np.save(join(crntg,"g%d.npy") %(numg),prelogitsg[0])
    #             numg+=1
    #             np.save(join(crntg,"g%d.npy") %(numg),prelogitsg[1])
    #             numg+=1
    #             np.save(join(crntg,"g%d.npy") %(numg),prelogitsg[2])
    #             numg+=1
    #             np.save(join(crntg,"g%d.npy") %(numg),prelogitsg[3])
    #             numg+=1
                    


merge_images(n_id,dataset_dir,dataset_dir_new)






