# -*- coding: utf-8 -*-
"""
Created on Sun May 14 21:25:41 2017

@author: Cherry
"""


#import os
#import scipy.misc
import numpy as np
import tensorflow as tf
import scipy.io as sio
from glob import glob
batch_size = 1
input_size = 400
feature_size = 1153
checkpoint_dir = './checkpoint'
matfn = './data/const_array.mat'
data1 = sio.loadmat(matfn)
const_array = data1['const_array']
const_array[1,31] = 0.3

#const_array = np.tile(np.linspace(0,0.3,32),[9,1])

def load_data():
    data = glob("./data/*.JPEG")
    return data

def lrelu(x,leaky = 0.03):
    return tf.maximum(x,x*leaky)    

def weights(shape,name1):
    weights_dic = load_weights();    
    if name1 in weights_dic:
        if name1=='wconv1_1':
            weights_value = np.mean(weights_dic[name1],axis = 2)
            weights_value.shape = 3,3,1,64
            w = tf.Variable(weights_value,name = name1,trainable = False)
        else:
            w = tf.Variable(weights_dic[name1],name = name1,trainable = False)
    else:
#        w = tf.Variable(tf.constant(np.sqrt(np.prod(shape)),"float32",shape),name = name1,trainable = True)
        w = tf.Variable(tf.random_normal(shape = shape,stddev= np.sqrt(2.0/shape[0])),name = name1,trainable = True)
    return w

def bias(shape,name2):    
    weights_dic = load_weights();
    if name2 in weights_dic:
        temp_b = weights_dic[name2]
        temp_b.shape = shape
        b = tf.Variable(temp_b,name = name2,trainable = False)
    else:
#        b = tf.Variable(tf.random_normal(shape = shape,stddev=0.02),name = name2,trainable = True)
        b = tf.Variable(tf.constant(0.0,"float32",shape),name = name2,trainable = True)
    return b

def deconv2d(input_, w, bia_,output_shape,strides=[1, 2, 2, 1]):
    deconv = tf.nn.conv2d_transpose(input_, w, output_shape,strides)
    deconv = tf.reshape(tf.nn.bias_add(deconv, bia_), deconv.get_shape())
    return deconv
    
def conv2d(input_,w,bia,strides=[1, 1, 1, 1],paddings = "SAME"):
    conv = tf.nn.conv2d(input_,w,strides,padding= paddings)
    return tf.nn.bias_add(conv,bia)
    
def load_weights():
    matfn = './data/vgg16_tf.mat'
    dic = sio.loadmat(matfn)
    return dic
    
def Bilinear(IM):
#    IM = IM/tf.reduce_sum(tf.reduce_sum(IM,axis=2,keep_dims=True),axis=1,keep_dims=True)
    output = tf.image.resize_bilinear(IM,size = [input_size,input_size],align_corners=True)
    return output
                     
def VGG16(X):
    #the first layer
    wconv1_1 = weights([3,3,1,64],name1 = 'wconv1_1')
#    wconv1_1 = tf.sum(wconv1_1,axis=2)
#    wconv1_1.shape = 3,3,1,64
    
    bconv1_1 = bias([64],name2 = 'bconv1_1')
    wconv1_2 = weights([3,3,64,64],name1 = 'wconv1_2')
    bconv1_2 = bias([64],name2 = 'bconv1_2')
    
    conv1_1 =  conv2d(X,wconv1_1,bconv1_1)
    conv1_2 =  conv2d(conv1_1,wconv1_2,bconv1_2)
    
    conv1_1 =  tf.nn.relu(conv1_1)
    conv1_2 =  tf.nn.relu(conv1_2)
    conv1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding= "SAME")
    
  #the second layer
    wconv2_1 = weights([3,3,64,128],name1 = 'wconv2_1')
    bconv2_1 = bias([128],name2 = 'bconv2_1')
    wconv2_2 = weights([3,3,128,128],name1 = 'wconv2_2')
    bconv2_2 = bias([128],name2 = 'bconv2_2')

    conv2_1 =  tf.nn.relu(conv2d(conv1,wconv2_1,bconv2_1))
    conv2_2 =  tf.nn.relu(conv2d(conv2_1,wconv2_2,bconv2_2))
    conv2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding= "SAME")
    
   
    wconv3_1 = weights([3,3,128,256],name1 = 'wconv3_1')
    bconv3_1 = bias([256],name2 = 'bconv3_1')
    wconv3_2 = weights([3,3,256,256],name1 = 'wconv3_2')
    bconv3_2 = bias([256],name2 = 'bconv3_2')
    wconv3_3 = weights([3,3,256,256],name1 = 'wconv3_3')
    bconv3_3 = bias([256],name2 = 'bconv3_3')

    conv3_1 =  tf.nn.relu(conv2d(conv2  ,wconv3_1,bconv3_1))
    conv3_2 =  tf.nn.relu(conv2d(conv3_1,wconv3_2,bconv3_2))
    conv3_3 =  tf.nn.relu(conv2d(conv3_2,wconv3_3,bconv3_3))
#    conv3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1],
#                        strides=[1, 2, 2, 1], padding= "SAME")
#    
#    #the 4th layer
#    wconv4_1 = weights([3,3,256,512],name1 = 'wconv4_1')
#    bconv4_1 = bias([512],name2 = 'bconv4_1')
#    wconv4_2 = weights([3,3,512,512],name1 = 'wconv4_2')
#    bconv4_2 = bias([512],name2 = 'bconv4_2')
#    wconv4_3 = weights([3,3,512,512],name1 = 'wconv4_3')
#    bconv4_3 = bias([512],name2 = 'bconv4_3')
#
#    conv4_1 =  tf.nn.relu(conv2d(conv3  ,wconv4_1,bconv4_1))
#    conv4_2 =  tf.nn.relu(conv2d(conv4_1,wconv4_2,bconv4_2))
#    conv4_3 =  tf.nn.relu(conv2d(conv4_2,wconv4_3,bconv4_3))
#    conv4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1],
#                        strides=[1, 2, 2, 1], padding= "SAME")
#                        
#    #the 5th layer
#    wconv5_1 = weights([3,3,512,512],name1 = 'wconv5_1')
#    bconv5_1 = bias([512],name2 = 'bconv5_1')
#    wconv5_2 = weights([3,3,512,512],name1 = 'wconv5_2')
#    bconv5_2 = bias([512],name2 = 'bconv5_2')
#    wconv5_3 = weights([3,3,512,512],name1 = 'wconv5_3')
#    bconv5_3 = bias([512],name2 = 'bconv5_3')
#
#    conv5_1 =  tf.nn.relu(conv2d(conv4  ,wconv5_1,bconv5_1))
#    conv5_2 =  tf.nn.relu(conv2d(conv5_1,wconv5_2,bconv5_2))
#    conv5_3 =  tf.nn.relu(conv2d(conv5_2,wconv5_3,bconv5_3))
#    conv5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1],
#                        strides=[1, 2, 2, 1], padding= "SAME")
#    conv5 = tf.reshape(conv5,[batch_size,-1])
    
    #the 6th layer
#    list_train_vgg16,wfc6 = weights([512*7*7,4096],name1 = 'wfc6',list_ = list_train_vgg16)
#    list_train_vgg16,bfc6 = bias([4096],name2 = 'bfc6',list_ = list_train_vgg16)
#
#    fc6 = conv2d(conv5,wfc6,bfc6, paddings = "VALID")    
#    
#    #the 7th layer
#    list_train_vgg16,wfc7 = weights([4096,4096],name1 = 'wfc7',list_ = list_train_vgg16)
#    list_train_vgg16,bfc7 = bias([4096],name2 = 'bfc7',list_ = list_train_vgg16)
#
#    fc7 = conv2d(fc6,wfc7,bfc7, paddings = "VALID")
#    
#    #the 8th layer
#    list_train_vgg16,wfc8 = weights([4096,1000],name1 = 'wfc8',list_ = list_train_vgg16)
#    list_train_vgg16,bfc8 = bias([1000],name2 = 'bfc8',list_ = list_train_vgg16)
#
#    fc8 = conv2d(fc7,wfc8,bfc8, paddings = "VALID")

    print "VGG16 has been loaded!"
    #auto-colorization
    H = tf.concat_v2(axis = 3,values = [X,Bilinear(conv1_1),Bilinear(conv1_2),Bilinear(conv2_1),Bilinear(conv2_2),Bilinear(conv3_1),Bilinear(conv3_2),Bilinear(conv3_3)])
#    H = tf.concat_v2(axis = 3,values = [X,Bilinear(conv1_1),Bilinear(conv1_2),Bilinear(conv2_1),Bilinear(conv2_2),Bilinear(conv3_1),Bilinear(conv3_2),Bilinear(conv3_3),Bilinear(conv4_1),Bilinear(conv4_2),Bilinear(conv4_3),Bilinear(conv5_1),Bilinear(conv5_2),Bilinear(conv5_3)])    
    return tf.reshape(H,[input_size*input_size,feature_size])
    
    
    
def T_prediction(H,hidden_size = 1024):
    #the 1st layer
    fc1_w = weights([feature_size,hidden_size*2], name1 = 'fc1_w')
    fc1_b = bias([hidden_size*2],name2 = 'fc1_b')
    hidden1 = tf.nn.relu(tf.matmul(H,fc1_w)+fc1_b)
    
    #the 2nd layer
    fc2_w = weights([hidden_size*2,hidden_size], name1 = 'fc2_w')
    fc2_b = bias([hidden_size],name2 = 'fc2_b')
    hidden2 = tf.nn.relu(tf.matmul(hidden1,fc2_w)+fc2_b)
    
    #the 3rd layerweights  
    fc3_w1_1   = weights([hidden_size,hidden_size/2],name1 = 'fc3_w1_1')
    fc3_b1_1   = bias([hidden_size/2],name2 = 'fc3_b1_1')
    fc3_w1_2   = weights([hidden_size/2,32],name1 = 'fc3_w1_2')
    fc3_b1_2   = bias([32],name2 = 'fc3_b1_2')
    hidden3_1 = tf.nn.relu(tf.matmul(hidden2,fc3_w1_1)+fc3_b1_1)
    output_X1 = tf.matmul(hidden3_1,fc3_w1_2)+fc3_b1_2
    
    fc3_w2_1   = weights([hidden_size,hidden_size/2],name1 = 'fc3_w2_1')
    fc3_b2_1   = bias([hidden_size/2],name2 = 'fc3_b2_1')
    fc3_w2_2   = weights([hidden_size/2,32],name1 = 'fc3_w2_2')
    fc3_b2_2   = bias([32],name2 = 'fc3_b2_2')
    hidden3_2 = tf.nn.relu(tf.matmul(hidden2,fc3_w2_1)+fc3_b2_1)
    output_X2 = tf.matmul(hidden3_2,fc3_w2_2)+fc3_b2_2
    
    fc3_w3_1   = weights([hidden_size,hidden_size/2],name1 = 'fc3_w3_1')
    fc3_b3_1   = bias([hidden_size/2],name2 = 'fc3_b3_1')
    fc3_w3_2   = weights([hidden_size/2,32],name1 = 'fc3_w3_2')
    fc3_b3_2   = bias([32],name2 = 'fc3_b3_2')
    hidden3_3 = tf.nn.relu(tf.matmul(hidden2,fc3_w3_1)+fc3_b3_1)
    output_X3 = tf.matmul(hidden3_3,fc3_w3_2)+fc3_b3_2
    
    fc3_w4_1   = weights([hidden_size,hidden_size/2],name1 = 'fc3_w4_1')
    fc3_b4_1   = bias([hidden_size/2],name2 = 'fc3_b4_1')
    fc3_w4_2   = weights([hidden_size/2,32],name1 = 'fc3_w4_2')
    fc3_b4_2   = bias([32],name2 = 'fc3_b4_2')
    hidden3_4 = tf.nn.relu(tf.matmul(hidden2,fc3_w4_1)+fc3_b4_1)
    output_X4 = tf.matmul(hidden3_4,fc3_w4_2)+fc3_b4_2
    
    fc3_w5_1   = weights([hidden_size,hidden_size/2],name1 = 'fc3_w5_1')
    fc3_b5_1   = bias([hidden_size/2],name2 = 'fc3_b5_1')
    fc3_w5_2   = weights([hidden_size/2,32],name1 = 'fc3_w5_2')
    fc3_b5_2   = bias([32],name2 = 'fc3_b5_2')
#    fc3_w5_3   = weights([32,32],name1 = 'fc3_w5_3')
#    fc3_b5_3   = bias([32],name2 = 'fc3_b5_3')
    hidden3_5 = tf.nn.relu(tf.matmul(hidden2,fc3_w5_1)+fc3_b5_1)
#    hidden3_5_2 = tf.nn.relu(tf.matmul(hidden3_5,fc3_w5_2)+fc3_b5_2)
    output_X5 = tf.matmul(hidden3_5,fc3_w5_2)+fc3_b5_2
    
    fc3_w6_1   = weights([hidden_size,hidden_size/2],name1 = 'fc3_w6_1')
    fc3_b6_1   = bias([hidden_size/2],name2 = 'fc3_b6_1')
    fc3_w6_2   = weights([hidden_size/2,32],name1 = 'fc3_w6_2')
    fc3_b6_2   = bias([32],name2 = 'fc3_b6_2')
#    fc3_w6_3   = weights([32,32],name1 = 'fc3_w6_3')
#    fc3_b6_3   = bias([32],name2 = 'fc3_b6_3')
    hidden3_6 = tf.nn.relu(tf.matmul(hidden2,fc3_w6_1)+fc3_b6_1)
#    hidden3_6_2 = tf.nn.relu(tf.matmul(hidden3_6,fc3_w6_2)+fc3_b6_2)
    output_X6 = tf.matmul(hidden3_6,fc3_w6_2)+fc3_b6_2
    
    fc3_w7_1   = weights([hidden_size,hidden_size/2],name1 = 'fc3_w7_1')
    fc3_b7_1   = bias([hidden_size/2],name2 = 'fc3_b7_1')
    fc3_w7_2   = weights([hidden_size/2,32],name1 = 'fc3_w7_2')
    fc3_b7_2   = bias([32],name2 = 'fc3_b7_2')
#    fc3_w7_3   = weights([32,32],name1 = 'fc3_w7_3')
#    fc3_b7_3   = bias([32],name2 = 'fc3_b7_3')
    hidden3_7 = tf.nn.relu(tf.matmul(hidden2,fc3_w7_1)+fc3_b7_1)
#    hidden3_7_2 = tf.nn.relu(tf.matmul(hidden3_7,fc3_w7_2)+fc3_b7_2)
    output_X7 = tf.matmul(hidden3_7,fc3_w7_2)+fc3_b7_2
    
    fc3_w8_1   = weights([hidden_size,hidden_size/2],name1 = 'fc3_w8_1')
    fc3_b8_1   = bias([hidden_size/2],name2 = 'fc3_b8_1')
    fc3_w8_2   = weights([hidden_size/2,32],name1 = 'fc3_w8_2')
    fc3_b8_2   = bias([32],name2 = 'fc3_b8_2')
#    fc3_w8_3   = weights([32,32],name1 = 'fc3_w8_3')
#    fc3_b8_3   = bias([32],name2 = 'fc3_b8_3')
    hidden3_8 = tf.nn.relu(tf.matmul(hidden2,fc3_w8_1)+fc3_b8_1)
#    hidden3_8_2 = tf.nn.relu(tf.matmul(hidden3_8,fc3_w8_2)+fc3_b8_2)
    output_X8 = tf.matmul(hidden3_8,fc3_w8_2)+fc3_b8_2
    
    fc3_w9_1   = weights([hidden_size,hidden_size/2],name1 = 'fc3_w9_1')
    fc3_b9_1   = bias([hidden_size/2],name2 = 'fc3_b9_1')
    fc3_w9_2   = weights([hidden_size/2,32],name1 = 'fc3_w9_2')
    fc3_b9_2   = bias([32],name2 = 'fc3_b9_2')
#    fc3_w9_3   = weights([32,32],name1 = 'fc3_w9_3')
#    fc3_b9_3   = bias([32],name2 = 'fc3_b9_3')
    hidden3_9 = tf.nn.relu(tf.matmul(hidden2,fc3_w9_1)+fc3_b9_1)
#    hidden3_9_2 = tf.nn.relu(tf.matmul(hidden3_9,fc3_w9_2)+fc3_b9_2)
    output_X9 = tf.matmul(hidden3_9,fc3_w9_2)+fc3_b9_2
#    return tf.concat_v2(axis=1,values=[output_X5,output_X6,output_X7,output_X8,output_X9]),\
#                tf.concat_v2(axis=1,values=[tf.nn.softmax(output_X5),tf.nn.softmax(output_X6),tf.nn.softmax(output_X7),\
#                tf.nn.softmax(output_X8),tf.nn.softmax(output_X9)])
    return tf.concat_v2(axis=1,values=[output_X1,output_X2,output_X3,output_X4,output_X5,output_X6,output_X7,output_X8,output_X9]),\
                tf.concat_v2(axis=1,values=[tf.nn.softmax(output_X1),tf.nn.softmax(output_X2),tf.nn.softmax(output_X3),\
                tf.nn.softmax(output_X4),tf.nn.softmax(output_X5),tf.nn.softmax(output_X6),tf.nn.softmax(output_X7),\
                tf.nn.softmax(output_X8),tf.nn.softmax(output_X9)])
    

       
def get_vectorised_T(IM):     
    temp = IM.size
    IM = np.reshape(IM,[temp/9,9])
    data_X  = np.zeros([temp/9,32*9])  
    for i in range(9):
        data_temp = np.reshape(const_array[i,:],[1,32])
        data_temp = np.tile(data_temp,[temp/9,1])
        data_temp = abs(np.transpose(np.tile(IM[:,i],[32,1]),[1,0])-data_temp)
        data_X[:,i*32:(i+1)*32] = np.equal(data_temp,np.transpose(np.tile(np.min(data_temp,axis = 1),[32,1]),[1,0]))
    return data_X


def inv_vetorization_T(data):
    data_V = np.zeros([input_size,input_size,9])
    data.shape = input_size,input_size,32*9
    for i in range(9):
        temp = np.tile(const_array[i,:],[input_size,input_size,1])
        data_V[:,:,i] = np.sum(data[:,:,i*32:(i+1)*32]*temp,2)
#        temp = const_array[i,:]
#        data_V[:,:,i] = temp[np.argmax(data[:,:,i*32:(i+1)*32],2)]                 
    return data_V
    
    
    
    