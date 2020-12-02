import numpy as np 
import cv2
import multiprocessing as mp
import os

def read_faces(path, label, min_num=1):
    
    data = np.load(path)
    imgs = np.transpose(data['colorImages'], (3,0,1,2))
    
    if imgs.shape[0]<min_num:
        return np.empty(0),np.empty(0)
    
    imgs = np.array([cv2.resize(img, (125,125)) for img in imgs])
    label = np.array([label for i in range(imgs.shape[0])])
    
    return imgs,label
    

def load_data(base_directory, min_num=1):

    """
    Reads all the images from the base_directory
    Uses multithreading to decrease the reading time drastically
    
    Args: 
    
    base_directory (str): The path to the top directory of the YouTube faces dataset
    min_num (int): The minimum number of images a class should to be included in the output
    
    Returns:
    
    datax (ndarray): numpy array of images 
    datay (ndarray): numpy array of labels
    
    """
    datax = None
    datay = None
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply(read_faces,
                          args=(
                              base_directory + '/' + file, '_'.join(file.split('_')[:-1]), min_num
                              )) for file in os.listdir(base_directory)]
    pool.close()
    
    for result in results:
        if len(result[0])==0:
            continue
        elif datax is None:
            datax = result[0]
            datay = result[1]
        else:
            datax = np.vstack([datax, result[0]])
            datay = np.concatenate([datay,result[1]])
            
    return datax, datay

def split_dataset(x,y,weight=0.5):
    
    """
    Splits the input dataset into training and testing sets
    
    Args: 
    
    x (ndarray): The input images
    y (ndarray): The array of labels
    weight (float): The ratio of the split. The training set will contain weight*(number of classes in the input dataset)
    
    Returns:
    
    trainx (ndarray): numpy array of training images 
    trainy (ndarray): numpy array of training labels
    testx (ndarray): numpy array of testing images 
    testy (ndarray): numpy array of testing labels
    
    """
    
    split = np.floor(weight*len(np.unique(y))).astype('int')
    _,indices = np.unique(y,return_index=True)
    idx = sorted(indices)[split]


    trainx,trainy = x[:idx],y[:idx]
    testx,testy = x[idx:],y[idx:]

    return trainx,trainy,testx,testy