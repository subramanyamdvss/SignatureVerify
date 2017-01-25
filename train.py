
import tensorflow as tf
import numpy as np
from random import shuffle
import os
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

work_path  = "/home/surya/SignatureVerify"
dataset_dir = 	"/home/surya/Documents/signature_eval"
dataset_dir_new = "/home/surya/Documents/signature_eval_features"

epochs = 100
iters = 1000
features_dir = "/home/surya/Documents/signature_features"
#ratio of fake and genuine images in training set
ratio=0.48628874
#select only even number for batch_size
batch_size=128


def create_fdirs(dataset_dir_f):
    
    """ Used to create directories for features """

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
create_fdirs(dataset_dir_new)

print "creating finished"

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
    
    numf=1
    numg=1

    # Load the model
    sess = tf.Session()
    arg_scope = inception_resnet_v2_arg_scope()
    with slim.arg_scope(arg_scope):
        input_tensor=tf.placeholder(tf.float32, (1,299,299,3), name=None)
        logits, end_points = inception_resnet_v2(input_tensor, is_training=False)
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_file)
    
    #used to produce forged comparisions
    for i in range(1,n_id+1):

        for j,f1 in enumerate(listdir(os.path.join(dataset_dir,"id_%d/genuine" %(i)))):

            for k,f2 in enumerate(listdir(os.path.join(dataset_dir,"id_%d/forge" %(i)))):
                # to balance the dataset
                if k>2*j :
                    break
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
    for i in range(1,n_id+1):

        for j,f1 in enumerate(listdir(os.path.join(dataset_dir,"id_%d/genuine" %(i)))):

            for k,f2 in enumerate(listdir(os.path.join(dataset_dir,"id_%d/genuine" %(i)))):
                
                f1p=os.path.join(dataset_dir,"id_%d/genuine" %(i))
                f1p=os.path.join(f1p,f1)
                f2p=os.path.join(dataset_dir,"id_%d/genuine" %(i))
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
                prelogitsg = creating_features(sess,logits,end_points,tmp,input_tensor)
                crntg = join(dataset_dir_new,"genuine")
                np.save(join(crntg,"g%d.npy") %(numg),prelogitsg[0])
                numg+=1
                np.save(join(crntg,"g%d.npy") %(numg),prelogitsg[1])
                numg+=1
                np.save(join(crntg,"g%d.npy") %(numg),prelogitsg[2])
                numg+=1
                np.save(join(crntg,"g%d.npy") %(numg),prelogitsg[3])
                numg+=1

def merge_images_eval(n_id,dataset_dir,dataset_dir_new):
    
    numf=1
    numg=1

    # Load the model
    sess = tf.Session()
    arg_scope = inception_resnet_v2_arg_scope()
    with slim.arg_scope(arg_scope):
        input_tensor=tf.placeholder(tf.float32, (1,299,299,3), name=None)
        logits, end_points = inception_resnet_v2(input_tensor, is_training=False)
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_file)
    
    #used to produce forged comparisions
    for i in range(1,n_id+1):

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
    for i in range(1,n_id+1):

        for j,f1 in enumerate(listdir(os.path.join(dataset_dir,"id_%d/genuine" %(i)))):

            for k,f2 in enumerate(listdir(os.path.join(dataset_dir,"id_%d/genuine" %(i)))):
                
                f1p=os.path.join(dataset_dir,"id_%d/genuine" %(i))
                f1p=os.path.join(f1p,f1)
                f2p=os.path.join(dataset_dir,"id_%d/genuine" %(i))
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
                prelogitsg = creating_features(sess,logits,end_points,tmp,input_tensor)
                crntg = join(dataset_dir_new,"genuine")
                np.save(join(crntg,"g%d.npy") %(numg),prelogitsg[0])
                numg+=1
                np.save(join(crntg,"g%d.npy") %(numg),prelogitsg[1])
                numg+=1
                np.save(join(crntg,"g%d.npy") %(numg),prelogitsg[2])
                numg+=1
                np.save(join(crntg,"g%d.npy") %(numg),prelogitsg[3])
                numg+=1
                    


merge_images_eval(100,dataset_dir,dataset_dir_new)
merge_images(38,"/home/surya/Documents/signature","/home/surya/Documents/signature_features")


def create_file_paths(features_dir):
	gen = join(features_dir,"genuine")
	genuine = listdir(gen)
	genuine = [(join(gen,genuine[i]),0) for i in xrange(len(genuine))]
	forg = join(features_dir,"forge")
	forge = listdir(forg)
	forge = [(join(forg,forge[i]),1) for i in xrange(len(forge))]
	return genuine,forge

def create_batch_list(genuine,forge):
	flag = True
	batch_list = []
	# print genuine
	shuffle(genuine)
	shuffle(forge)
	for i in xrange(31636-(batch_size//2)):
		c = genuine[i:i+batch_size//2]+forge[i:i+batch_size//2]
		shuffle(c)
		batch_list.append(c)
		i+=batch_size//2-1
	return batch_list

genuine,forge = create_file_paths(features_dir)

# batch_list = create_batch_list(genuine,forge)


def create_batch(batch_list):
	batch = batch_list[0]
	batch_list[1:len(batch_list)]
	shuffle(batch)
	batch = [(np.squeeze(np.load(batch[i][0])),batch[i][1]) for i in xrange(len(batch))]
	batchx = [batch[i][0] for i in xrange(len(batch))]
	batchy = [batch[i][1] for i in xrange(len(batch))]
	batchy_ = np.zeros((len(batch),2))
	for i in len(batch):
		batchy_[i][batchy[i]]=1 
	batchx = np.stack(batchx , axis = 0)
	batchy=batchy_
	return batchx , batchy ,batch_list
	

def eval(gs,flag = False):

	x = tf.placeholder(tf.float32, [1, 1536])
	y_ = tf.placeholder(tf.float32, [1, 2])
	y2 = model(x)
	y3 = tf.nn.softmax(y2)
	genuine , forge = create_file_paths(dataset_dir_new)
	tp = 0
	tn = 0
	fp = 0
	fn = 0
	acc = 0
	totalg = 0
	totalf = 0
	total = 0
	for fea in genuine:
		c = np.load(fea)[0]
		batchy_ = np.zeros((1,2))
		batchy = np.load(fea)[1]
		batchy_[batchy]=1
		totalg+=1
		total+=1
		logits = sess.run(y3,feed_dict={x:c,y_:batchy_})
		if np.argmax(y3) == 0 and np.argmax(y_) == 0:
			tp+=1
		if np.argmax(y3) == 0 and np.argmax(y_) == 1:
			fp+=1
		if np.argmax(y3) == 1 and np.argmax(y_) == 0:
			fn+=1
		if np.argmax(y3) == 1 and np.argmax(y_) == 1:
			tn+=1
	acc = (tp+tn)/total
	pg = tp/(tp+fp)
	rg = tp/(tp+fn)
	fg = 2*pg*rg/(pg+rg)
	pf = tn/(tn+fn)
	rf = tn/(tn+fp)
	ff = 2*pf*rf/(pf+rf)
	if flag:
		print " totalg:%f totalf:%f global step : %f accuracy : %f  fscore genuine : %f fscore forge: %f " %(totalg,totalf,gs,acc,fg,ff),(pg,rg,pf,rf)
	else:
		print " global step : %f accuracy : %f  fscore genuine : %f fscore forge: %f " %(gs,acc,fg,ff),(pg,rg,pf,rf)

def model(x):
	 # Create the model
	
	
  	W1 = tf.Variable(tf.zeros([1536, 1000]))
 	b1 = tf.Variable(tf.zeros([1000]))
  	y1 = tf.nn.relu(tf.matmul(x, W) + b1)
  	
  	W2 = tf.Variable(tf.zeros([1000, 2]))
  	b2 = tf.Variable(tf.zeros([2]))
  	y2 = tf.matmul(y1,W2)+b2

  	return y2 



def train(flag = True):
	x = tf.placeholder(tf.float32, [None, 1536])
	y_ = tf.placeholder(tf.float32, [None, 2])
	y2 = model()

  	cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y2))
  	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  	# Add ops to save and restore all the variables.
	saver = tf.train.Saver()

  	sess = tf.InteractiveSession()
  	tf.global_variables_initializer().run()
    # Train
  	for ep in range(epochs):
  		batch_list = create_batch_list(genuine,forge)
  		shuffle(batch_list)
  		for it in range(iters):
  			if len(batch_list)==0:
  				break
    		batchx, batchy , batch_list = create_batch(batch_list)
    		sess.run(train_step, feed_dict={x: batchx, y_: batchy})
    		if ep*it % 100 :
    			eval()
    		if ep*it %1000 :
    			if flag:
    				eval(ep*it ,flag)
    				flag=False
    				saver.save(sess, work_path,global_step=ep*it)
    			else:
    				saver.save(sess, work_path,global_step=ep*it)
    				eval(ep*it)	
    	saver.save(sess, work_path,global_step=ep*it)
    	eval(ep*it)	

train(True)
























