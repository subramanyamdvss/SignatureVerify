import tensorflow as tf
import cv2
slim = tf.contrib.slim
from PIL import Image
from create_and_fill import *
from inception_resnet_v2 import *
import numpy as np
from os import listdir
from os.path import isfile, join

# path to your inception checkpoint file
checkpoint_file = 'home/surya/Documents/inception_resnet_v2_2016_08_30.ckpt'
# path to your dataset
dataset_dir="/home/surya/Documents/signature"
# path to your new dataset for features.
dataset_dir_new="/home/surya/Documents/signature_features"
# call to create new dataset
create_dirs(38,dataset_dir_new)



# sample_images = ['dog.jpg', 'panda.jpg']
#Load the model
# sess = tf.Session()
# arg_scope = inception_resnet_v2_arg_scope()
# with slim.arg_scope(arg_scope):
#   logits, end_points = inception_resnet_v2(input_tensor, is_training=False)
# saver = tf.train.Saver()
# saver.restore(sess, checkpoint_file)
# for image in sample_images:
#   im = Image.open(image).resize((299,299))
#   im = np.array(im)
#   im = im.reshape(-1,299,299,3)
#   predict_values, logit_values = sess.run([end_points['Predictions'], logits], feed_dict={input_tensor: im})
#   print (np.max(predict_values), np.max(logit_values))
#   print (np.argmax(predict_values), np.argmax(logit_values))

def creating_features(flag = True):
    for i in range(1,39):

        for f1 in listdir(os.path.join(dataset_dir,"id_%d/genuine" %(i))):

            for f2 in listdir(os.path.join(dataset_dir,"id_%d/forge" %(i))):
                f1p=os.path.join(dataset_dir,"id_%d/genuine" %(i))
                f1p=os.path.join(f1p,f1)
                f2p=os.path.join(dataset_dir,"id_%d/forge" %(i))
                f2p=os.path.join(f2p,f2)
                # ftens1=tf.image.decode_jpeg(f1p,channels=1)
                # ftens2=tf.image.decode_jpeg(f2p,channels=1)
                im1 = cv2.imread(f1p)
                im2 = cv2.imread(f2p)
                im1gr = cv2.resize(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY),(299,299))
                im2gr = cv2.resize(cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY),(299,299))
                if i==1:
                    tmp=np.dstack((im1gr,im2gr))
                    if  flag:
                        cv2.imshow("dvss",im1gr)
                        flag=False
                        print tmp.shape
                    


creating_features()






