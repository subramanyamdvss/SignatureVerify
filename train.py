
import tensorflow as tf
import numpy as np
from random import shuffle
import os
import time
from inception_preprocessing import *
from inception_resnet_v2 import *
import numpy as np
from os import listdir
from os.path import isfile, join
import tensorflow as tf
import cv2
slim = tf.contrib.slim
from shutil import copy,rmtree

checkpoint_file = '/home/surya/Documents/inception_resnet_v2_2016_08_30.ckpt'

work_path  = "/home/surya/SignatureVerify/ckpt-dir"
dataset_dir =   "/home/surya/Documents/signature_eval"
dataset_dir_new = "/home/surya/Documents/signature_eval_features"

epochs =20000
iters = 1000
features_dir = "/home/surya/Documents/signature_features"
#ratio of fake and genuine images in training set
ratio=0.48628874
#select only even number for batch_size
batch_size=128


def create_fdirs(dataset_dir_f):
    
    """ Used to create directories for features """
    print "IN create_fdirs FUNC"
    mypath = dataset_dir_f
    direcs = [join(mypath,f) for f in listdir(mypath)]
    for dr in direcs:
        rmtree(dr)
    crntg = os.path.join(mypath,"genuine")
    crntf = os.path.join(mypath,"forge")
    if not os.path.exists(crntg):
        os.makedirs(crntg)
    if not os.path.exists(crntf):
        os.makedirs(crntf)
    return



#to merge images of evaluation dataset and pass it through inception-resnet-v2
# create_fdirs(dataset_dir_new)

# print "creating finished"

# def creating_features(sess,logits,end_points,tmp,input_tensor):
#     print "IN creating_features FUNC"

#     prelog=[]
#     for i,im in enumerate(tmp):
#         im = [im]
#         im = np.array(im)
#         print type(im)
#         prelogits,logits_aft= sess.run([end_points['PreLogitsFlatten'],logits],feed_dict={input_tensor:im})
#         prelog.append(prelogits)
#     return prelog



# def merge_images_eval(n_id,dataset_dir,dataset_dir_new):
#     print "IN merge_images_eval FUNC"
#     numf=1
#     numg=1

#     # Load the model
#     sess = tf.Session()
#     arg_scope = inception_resnet_v2_arg_scope()
#     with slim.arg_scope(arg_scope):
#         input_tensor=tf.placeholder(tf.float32, (1,299,299,3), name=None)
#         logits, end_points = inception_resnet_v2(input_tensor, is_training=False)
#     saver = tf.train.Saver()
#     saver.restore(sess, checkpoint_file)
    
#     #used to produce forged comparisions
#     for i in range(1,n_id+1):

#         for j,f1 in enumerate(listdir(os.path.join(dataset_dir,"id_%d/genuine" %(i)))):

#             for k,f2 in enumerate(listdir(os.path.join(dataset_dir,"id_%d/forge" %(i)))):
                
#                 f1p=os.path.join(dataset_dir,"id_%d/genuine" %(i))
#                 f1p=os.path.join(f1p,f1)
#                 f2p=os.path.join(dataset_dir,"id_%d/forge" %(i))
#                 f2p=os.path.join(f2p,f2)
#                 im1 = cv2.imread(f1p)
#                 im2 = cv2.imread(f2p)
#                 im1gr = cv2.resize(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY),(299,299))
#                 im2gr = cv2.resize(cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY),(299,299))
#                 tmp1=np.dstack((im1gr,im2gr,np.zeros((299,299))))
#                 tmp1 = 2*(tmp1/255.0)-1.0
                
#                 # images which are flipped left  right , up  down 
#                 tmp1 = np.dstack((im1gr,im2gr,np.zeros((299,299))))
#                 tmp2 = np.flipud(tmp1)
#                 tmp3 = np.fliplr(tmp1)
#                 tmp4 = np.fliplr(tmp2)
#                 tmp=[tmp1,tmp2,tmp3,tmp4]
#                 prelogitsf = creating_features(sess,logits,end_points,tmp,input_tensor)
#                 crntf = join(dataset_dir_new,"forge")
#                 np.save(join(crntf,"f%d.npy") %(numf),prelogitsf[0])
#                 numf+=1
#                 np.save(join(crntf,"f%d.npy") %(numf),prelogitsf[1])
#                 numf+=1
#                 np.save(join(crntf,"f%d.npy") %(numf),prelogitsf[2])
#                 numf+=1
#                 np.save(join(crntf,"f%d.npy") %(numf),prelogitsf[3])
#                 numf+=1

#     # used to produce genuine comparisions
#     for i in range(1,n_id+1):

#         for j,f1 in enumerate(listdir(os.path.join(dataset_dir,"id_%d/genuine" %(i)))):

#             for k,f2 in enumerate(listdir(os.path.join(dataset_dir,"id_%d/genuine" %(i)))):
                
#                 f1p=os.path.join(dataset_dir,"id_%d/genuine" %(i))
#                 f1p=os.path.join(f1p,f1)
#                 f2p=os.path.join(dataset_dir,"id_%d/genuine" %(i))
#                 f2p=os.path.join(f2p,f2)
#                 im1 = cv2.imread(f1p)
#                 im2 = cv2.imread(f2p)
#                 im1gr = cv2.resize(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY),(299,299))
#                 im2gr = cv2.resize(cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY),(299,299))
#                 tmp1=np.dstack((im1gr,im2gr,np.zeros((299,299))))
#                 tmp1 = 2*(tmp1/255.0)-1.0
                
#                 # images which are flipped left  right , up  down 
#                 tmp1 = np.dstack((im1gr,im2gr,np.zeros((299,299))))
#                 tmp2 = np.flipud(tmp1)
#                 tmp3 = np.fliplr(tmp1)
#                 tmp4 = np.fliplr(tmp2)
#                 tmp=[tmp1,tmp2,tmp3,tmp4]
#                 prelogitsg = creating_features(sess,logits,end_points,tmp,input_tensor)
#                 crntg = join(dataset_dir_new,"genuine")
#                 np.save(join(crntg,"g%d.npy") %(numg),prelogitsg[0])
#                 numg+=1
#                 np.save(join(crntg,"g%d.npy") %(numg),prelogitsg[1])
#                 numg+=1
#                 np.save(join(crntg,"g%d.npy") %(numg),prelogitsg[2])
#                 numg+=1
#                 np.save(join(crntg,"g%d.npy") %(numg),prelogitsg[3])
#                 numg+=1
                    


# merge_images_eval(100,dataset_dir,dataset_dir_new)


def create_file_paths(f_dir):
    print "IN create_file_paths FUNC"
    gen = join(f_dir,"genuine")
    genuine = sorted(listdir(gen))
    forg = join(f_dir,"forge")
    forge = sorted(listdir(forg))
    genuine = [(join(gen,genuine[i]),0) for i in xrange(len(genuine))]
    forge = [(join(forg,forge[i]),1) for i in xrange(len(genuine))]
    print "genuine length ",len(genuine)
    assert (len(genuine))
    return genuine,forge

def create_batch_list(genuine,forge):
    print "IN create_batch_list FUNC"
    flag = True
    batch_list = []
    # print genuine
    shuffle(genuine)
    shuffle(forge)
    for i in xrange(0,46184-(batch_size//2),batch_size//2):
        c = genuine[i:i+batch_size//2]+forge[i:i+batch_size//2]
        shuffle(c)
        if len(c)==0:
            
            continue

        batch_list.append(c)
            
    
    return batch_list

# genuinet , forget = create_file_paths(dataset_dir_new)
genuine , forge = create_file_paths(features_dir)
# batch_list = create_batch_list(genuine,forge)


def create_batch(batch_list):
    # print "IN create_batch FUNC"
    batch = batch_list[0]
    batch_list=batch_list[1:len(batch_list)]
    shuffle(batch)
    batch = [(np.squeeze(np.load(batch[i][0])),batch[i][1]) for i in xrange(len(batch))]
    batchx = [batch[i][0] for i in xrange(len(batch))]
    batchy = [batch[i][1] for i in xrange(len(batch))]
    batchy_ = np.zeros((len(batch),1))
    for i in xrange(len(batch)):
        if batchy[i]==1:
            batchy_[i]=np.array([1]) 
    # print len(batch)
    assert (len(batch))
    batchx = np.stack(batchx , axis = 0)
    batchy=batchy_
    return batchx , batchy ,batch_list
    

def eval(batcher,loss,sess,y4,x,y_, gs,flag = False):

    print "IN EVAL FUNC"
    
    
    # nw = genuine + forge
    # nwt = genuinet + forget
    tp = 0.0+1
    tn = 0.0+1
    fp = 0.0+1
    fn = 0.0+1
    acc = 0.0
    totalg = 0.0
    totalf = 0.0
    total = 0.0
    for fea in batcher:
        batchy=fea[1]
        c=fea[0]
        batchy_ = np.zeros((1,2))
        batchy_[0][batchy]=1
        if batchy==0:
            totalg+=1
        else:
            totalf+=1
        total+=1
        logits = sess.run(y4,feed_dict={x:c,y_:batchy_})
        if np.argmax(logits) == 0 and np.argmax(batchy_) == 0:
            tp+=1
        if np.argmax(logits) == 0 and np.argmax(batchy_) == 1:
            fp+=1
        if np.argmax(logits) == 1 and np.argmax(batchy_) == 0:
            fn+=1
        if np.argmax(logits) == 1 and np.argmax(batchy_) == 1:
            tn+=1
    acc = (tp+tn)/total
    pg = tp/(tp+fp)
    rg = tp/(tp+fn)
    fg = 2*pg*rg/(pg+rg)
    pf = tn/(tn+fn)
    rf = tn/(tn+fp)
    ff = 2*pf*rf/(pf+rf)
    if flag:
        print "TRAINING METRICS :( loss:%f totalg:%f totalf:%f global step : %f accuracy : %f  fscore genuine : %f fscore forge: %f) " %(loss,totalg,totalf,gs,acc,fg,ff),(tp,fp,fn,tn)
    else:
        print "TRAINING METRICS :( loss:%f global step : %f accuracy : %f  fscore genuine : %f fscore forge: %f) " %(loss,gs,acc,fg,ff),(tp,fp,fn,tn)
    # tp = 1.0
    # tn = 1.0
    # fp = 1.0
    # fn = 1.0
    # acc = 0.0
    # totalg = 0.0
    # totalf = 0.0
    # total = 0.0
    # for fea in nw:
    #     c = np.load(fea[0])
    #     batchy_ = np.zeros((1,2))
    #     batchy = fea[1]
    #     batchy_[0][batchy]=1
    #     if batchy==0:
    #         totalg+=1
    #     else:
    #         totalf+=1
    #     total+=1
    #     logits = sess.run(y4,feed_dict={x:c,y_:batchy_})
    #     if np.argmax(logits) == 0 and np.argmax(batchy_) == 0:
    #         tp+=1
    #     if np.argmax(logits) == 0 and np.argmax(batchy_) == 1:
    #         fp+=1
    #     if np.argmax(logits) == 1 and np.argmax(batchy_) == 0:
    #         fn+=1
    #     if np.argmax(logits) == 1 and np.argmax(batchy_) == 1:
    #         tn+=1
    # acc = (tp+tn)/total
    # pg = tp/(tp+fp)
    # rg = tp/(tp+fn)
    # fg = 2*pg*rg/(pg+rg)
    # pf = tn/(tn+fn)
    # rf = tn/(tn+fp)
    # ff = 2*pf*rf/(pf+rf)
    # if flag:
    #     print "EVALUATION METRICS :( loss:%f totalg:%f totalf:%f global step : %f accuracy : %f  fscore genuine : %f fscore forge: %f) " %(loss,totalg,totalf,gs,acc,fg,ff),(tp,fp,fn,tn)
    # else:
    #     print "EVALUATION METRICS :( loss:%f global step : %f accuracy : %f  fscore genuine : %f fscore forge: %f) " %(loss,gs,acc,fg,ff),(tp,fp,fn,tn)
    return


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def model(x):
     # Create the model
    print "IN model FUNC"
    
    W1 = weight_variable([1536, 1000])
    b1 = bias_variable([1000])
    
    y1 =tf.nn.relu(tf.matmul(x,W1)+b1) 
    W2 = weight_variable([ 1000,1])
    b2 = bias_variable([1])
    y2 = tf.matmul(y1,W2)+b2 
    return y2 

def evalt(batcherx,batchery,loss,sess,y4,x,y_, gs,flag = False):    
    
    correct_prediction=tf.equal(y4,y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc = sess.run(accuracy, feed_dict={x:np.stack(batcherx),y_:np.stack(batchery)})
    print "training accuracy: %f loss: %f" %(acc,loss)

def main():
    flag = True
    print "IN train FUNC"
    x = tf.placeholder(tf.float32, [None, 1536])
    y_ = tf.placeholder(tf.float32, [None, 1])
    y2 = model(x)
    y4 = tf.sigmoid(y2)
    cross_entropy = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(y2,y_ ))
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()
    batch_list = create_batch_list(genuine,forge)
    nw = genuine + forge
    print  'batchery'
    batcherx=[np.squeeze(np.load(fl1)) for fl1,fl2 in nw]
    batchery=[np.array([fl2]) for fl1,fl2 in nw]

    print "end batchery"
    
    # Train
    for ep in range(epochs):
        tm= time.time()
        print "batch_list : ",len(batch_list)
        batch_l= batch_list
        shuffle(batch_l)
        for it in range(20*iters):
            if len(batch_l)==0:
                break
            ln = len(batch_l)
            batchx, batchy , batch_l = create_batch(batch_l)
            assert (ln-1==len(batch_l))
            # print len(batchy), len(batchx)
            # assert (len(batchx)==128 and len(batchy)==128)
            
            _ , loss,logits = sess.run([train_step,cross_entropy, y4], feed_dict={x: batchx, y_: batchy})
            
            if ((it+1)) % 100 ==0 :
                print "epoch: %d  iteration : %d" %(ep+1,(it+1))
        duration = time.time()-tm

        print "time taken for the epoch to complete: ",duration  
        
        if True:           
            if ep%10==0:
                saver.save(sess, join(work_path,"model_e%d_it%d.ckpt" %((ep+1),(it+1))))
            
                evalt(batcherx,batchery,loss,sess,y4,x,y_,494*(ep+1)+(it+1))    
        if True:
            if ep%1==0:
                saver.save(sess, join(work_path,"model_e%d_it%d.ckpt" %((ep+1),(it+1))))
            
                    

if __name__ == '__main__':
    main()
























