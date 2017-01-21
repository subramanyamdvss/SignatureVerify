
import os
from shutil import copy,rmtree
from os import listdir
from os.path import isfile, join


#where you want to create the dataset

dataset_dir="/home/surya/Documents/signature"
dataset_train_dutch_g="/home/surya/Documents/sigcom11/OfflineSignatures/Dutch/TrainingSet/Offline Genuine"
dataset_train_dutch_f="/home/surya/Documents/sigcom11/OfflineSignatures/Dutch/TrainingSet/Offline Forgeries"
dataset_train_chinese_g="/home/surya/Documents/sigcom11/OfflineSignatures/Chinese/TrainingSet/Offline Genuine"
dataset_train_chinese_f="/home/surya/Documents/sigcom11/OfflineSignatures/Chinese/TrainingSet/Offline Forgeries"
dataset_train_09="/home/surya/Documents/sigcom09/SigComp2009-training/NISDCC-offline-all-001-051-6g"
# directories for features
dataset_dir_f="/home/surya/Documents/signature_features"

def create_dirs(dir_n, dataset_dir):

    """  Used to create empty directories   """
    mypath=dataset_dir
    direcs = [join(mypath,f) for f in listdir(mypath)]
    for dr in direcs:
        rmtree(dr)
    dir = ["id_%d" %(i+1) for i in range(dir_n)]
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    for direc in dir:
        if not os.path.exists(os.path.join(dataset_dir,direc)):
            os.makedirs(os.path.join(dataset_dir,direc))
            crnt = os.path.join(dataset_dir,direc)
            crntg = os.path.join(crnt,"genuine")
            crntf = os.path.join(crnt,"forge")
            if not os.path.exists(crntg):
                os.makedirs(crntg)
            if not os.path.exists(crntf):
                os.makedirs(crntf)
    return

def create_fdirs():
    
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

#fills only 16 dutch and 10 chinese  ids from 2011 and 12 ids from 2009
def fill_dir_train():
    """fill the training data in the directories created"""

#for dutch train genuine
    mypath=dataset_train_dutch_g
    onlyfiles = [(f,join(mypath,f)) for f in listdir(mypath) if isfile(join(mypath, f))]
    for f,fpath in onlyfiles:
        dcd = int(f[0:3])
        copy(fpath,join(dataset_dir,("id_%d/genuine") %(dcd)))
#for dutch train forge
    mypath=dataset_train_dutch_f
    onlyfiles = [(f,join(mypath,f)) for f in listdir(mypath) if isfile(join(mypath, f))]
    for f,fpath in onlyfiles:
        dcd = int(f[4:7])
        copy(fpath,join(dataset_dir,("id_%d/forge") %(dcd)))
#for chinese train genuine
    mypath=dataset_train_chinese_g
    onlyfiles = [(f,join(mypath,f)) for f in listdir(mypath) if isfile(join(mypath, f))]
    for f,fpath in onlyfiles:
        dcd = int(f[0:3])
        copy(fpath,join(dataset_dir,("id_%d/genuine") %(16+dcd)))
#for chinese train forge
    mypath=dataset_train_chinese_f
    onlyfiles = [(f,join(mypath,f)) for f in listdir(mypath) if isfile(join(mypath, f))]
    for f,fpath in onlyfiles:
        dcd = int(f[4:7])
        copy(fpath,join(dataset_dir,("id_%d/forge") %(16+dcd)))
#for dataset train 09
    mypath=dataset_train_09
    onlyfiles = [(f,join(mypath,f)) for f in listdir(mypath) if isfile(join(mypath, f))]
    for f,fpath in onlyfiles:
        dcd = int(f[7:10])
        if dcd<=12:
            copy(fpath,join(dataset_dir,("id_%d/genuine") %(26+dcd)))
        else:
            dcm = int(f[11:14])
            copy(fpath,join(dataset_dir,("id_%d/forge") %(26+dcm)))


#############incomplete
def fill_dir_eval(src):
    mypath=src
    onlyfiles = [(f,join(mypath,f)) for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles=sorted(onlyfiles)
    print (int(onlyfiles[1][0][4:7]))

    # for f in onlyfiles:
    #     tmp=int(f[4:7])
    #     if i!=tmp:
    #         while i!=tmp:
    #             missingid.append(i)
    #             i+=1

    #     if i==tmp:
    #         i+=1
    # if len(missingid)==940:
    #     print ("you won")
    # else:
    #     print ("you lose")


# dataset_eval="/home/surya/Documents/sigcom09/SigComp2009-evaluation/genuines"
# fill_dir_eval(dataset_eval)
create_dirs(38, dataset_dir)
fill_dir_train()
















