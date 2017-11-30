# -*- coding: utf-8 -*-
"""
Created on Sun May 14 19:38:23 2017

@author: Cherry
"""


## by Qian Song on Feb. 26th 2017


import utils_SD3
import numpy as np
import time
import scipy.io as sio
import h5py
import tensorflow as tf
import random

class Pol_SD(object):
    def __init__(self, sess):
        self.sess = sess
        
        self.learning_rate = 0.0001
#        self.beta1 = 0.9
#        self.beta2 = 0.999
#        self.batch_size = 1
#        self.lamda = 0.1
#        self.d_h = 2
#        self.d_w = 2
        self.output_size = 400
        self.training_size = 19
        self.total_size = 189
        self.feature_size = 1153
        self.test_size = 4
        self.re_total_size = 135

        self.model_build()
        
    def model_build(self):
        ##Build Model
        self.VVVV = tf.placeholder(tf.float32,[None,self.output_size,self.output_size,1])
        self.H = tf.placeholder(tf.float32,[None,self.feature_size])
        self.X_true = tf.placeholder(tf.float32,[None,32*9])
        
        self.hypercolumn = utils_SD3.VGG16(self.VVVV)
        self.X,self.X_ = utils_SD3.T_prediction(self.H)
        
        
#        self.d_loss = tf.reduce_mean(((self.X_true - self.X)**2)/2.0)
#        self.d_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = self.X_true, logits = self.X))
        self.d_loss = - tf.reduce_mean(self.X_true*tf.log(self.X_+1e-7))
        self.optim = tf.train.AdamOptimizer(learning_rate = self.learning_rate, epsilon = 1e-6) \
                                  .minimize(self.d_loss)
        self.saver = tf.train.Saver()

    def train(self):
        init = tf.initialize_all_variables()
        self.sess.run(init)
        start_time = time.time()
        matfn = './samples/SD/mean_and_var.mat'
        data1 = sio.loadmat(matfn)
        temp_mean = data1['mean']
        temp_mean = np.mean(temp_mean,axis=0)
        temp_var  = data1['var']
        temp_var[temp_var<1.0] = 1.0
        temp_var = np.mean(temp_var,axis=0)
        
        print("[*]Loading Model...")
        self.saver.restore(self.sess, "./checkpoint/SD_224_224/Generate model-SD16")
        print("[*]Load successfully!")
        

#        self.is_train = True
#        data_X,data_V = self.load_data()
#
#        #Training steps:=====================================
#        counter = 0
#        temp_list1 = np.linspace(0,self.output_size*self.output_size-1,self.output_size*self.output_size,dtype = 'int')
#        temp_list2 = np.linspace(0,self.training_size-1,self.training_size,dtype = 'int')
#        for epoch in range(100):
#            batch_idxs = len(data_X)
#            random.shuffle(temp_list2)
#            random.shuffle(temp_list1)
#            for idx in temp_list2:                
#                batch_V = np.reshape(data_V[idx,:,:],[1,self.output_size,self.output_size,1])
#                temp_H  = self.sess.run([self.hypercolumn],feed_dict = {self.VVVV:batch_V})
#                temp_H = temp_H[0]
#                temp_H = (temp_H-temp_mean)/temp_var                
#                temp_X = utils_SD3.get_vectorised_T(data_X[idx,:,:,:])
#                
#                for index in range(80):                                       
#                    batch_H = temp_H[temp_list1[index*2000:(index+1)*2000],:]
#                    batch_X = temp_X[temp_list1[index*2000:(index+1)*2000],:]                
##                    X,hd3,w,b = self.sess.run([self.X_,self.h3,self.fc3_w1_2,self.fc3_b1_2], feed_dict={self.H: temp_H})
#                    loss1,train_step,X = self.sess.run([self.d_loss, self.optim,self.X_], feed_dict={self.H: batch_H, self.X_true:batch_X})
#                
#                    counter += 1
#                    if np.mod(counter,10)==9:
#                        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f" \
#                                    % (epoch, idx+1, batch_idxs,
#                                        time.time() - start_time,loss1))                                          
#                     
##                    if np.mod(counter,100) == 99:
##                        print X[1,0:32]
##                        print batch_X[1,0:32]
#            self.saver.save(self.sess,"./checkpoint/SD_224_224/Generate model-SD16")
#            print("[*]Save Model...")
#            
##            self.cor_rate = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.X_,1),tf.argmax(self.X_true,1)),"float32"))
##            print self.sess.run(self.cor_rate,feed_dict={self.H: temp_H,self.X_true:temp_X[:,32*4:32*5]})
##            for i in range(9):
##                self.cor_rate = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.X_[:,32*i:32*(i+1)],1),tf.argmax(self.X_true[:,32*i:32*(i+1)],1)),"float32"))
##                print self.sess.run(self.cor_rate,feed_dict={self.H: temp_H,self.X_true:temp_X})
##                
#                
#            temp_X.shape = 400,400,32*9
#            data_true = utils_SD3.inv_vetorization_T(temp_X)
#            
##            X_temp = np.ones([400,400,32*9])
#            X = self.sess.run([self.X_], feed_dict={self.H: temp_H})
##            X,hd3,w,b = self.sess.run([self.X_,self.h3,self.fc3_w1_2,self.fc3_b1_2], feed_dict={self.H: temp_H})
#            X = X[0]
#            X.shape = 400,400,32*9
##            X_temp[:,:,32*4:] = X
#            data_generated = utils_SD3.inv_vetorization_T(X)
##            sio.savemat('data_G.mat',{'G':data_generated})
#            
#            for i in range(3):
#                print np.corrcoef((data_true[:,:,i]).flatten(),(data_generated[:,:,i]).flatten())
#                
#            for i in range(6):
#                print np.corrcoef((data_true[:,:,i+3]+1.0).flatten(),(data_generated[:,:,i+3]+1.0).flatten())      
#        
        self.is_train = False
        data_X,data_V,test_V = self.load_data()
        
        
            
#        Re_data = np.zeros([self.re_total_size,self.output_size,self.output_size,9])
#        for i in range(self.re_total_size):
#            batch_V = np.reshape(data_V[i,:,:],[1,self.output_size,self.output_size,1])
#            batch_H = self.sess.run([self.hypercolumn],feed_dict={self.VVVV:batch_V})
#            batch_H = batch_H[0]
#            batch_H = (batch_H-temp_mean)/temp_var
#            val_X   = self.sess.run([self.X_],feed_dict={self.H: batch_H})
#            Re_data[i,:,:,:] = utils_SD3.inv_vetorization_T(val_X[0])
#        sio.savemat('./samples/SD/test_SD2_re0526.mat',{'Re_data':Re_data})
#        
        
#        temp_X = utils_SD3.get_vectorised_T(data_X)
#        data_q = np.zeros([self.total_size,self.output_size,self.output_size,9])
#        for i in range(self.total_size):
#            data_q[i,:,:,:] = utils_SD3.inv_vetorization_T(temp_X[i*self.output_size*self.output_size:(i+1)*self.output_size*self.output_size,:])
#        sio.savemat('./samples/SD/SD1_quan.mat',{'data_q':data_q})
#        
#        print np.mean(abs(Re_data - data_q))


#        #Test data:========================================
        data_test = np.zeros([self.test_size,self.output_size,self.output_size,9])
        for i in range(self.test_size):
            test_batch_V = np.reshape(test_V[i,:,:],[1,self.output_size,self.output_size,1])
            batch_H = self.sess.run([self.hypercolumn],feed_dict={self.VVVV:test_batch_V})
            batch_H = batch_H[0]
            batch_H = (batch_H-temp_mean)/temp_var
            test_X   = self.sess.run([self.X_],feed_dict={self.H: batch_H})
            data_test[i,:,:,:] = utils_SD3.inv_vetorization_T(test_X[0])
        sio.savemat('./samples/SD/test_MN150605_VVVV_re.mat',{'Re_data':data_test})
        

    def load_data(self):
        matfn = './data/train_SD1109.mat'
        data1 = h5py.File(matfn,'r')
        data = data1['data']           
        data = np.transpose(data,axes = [3,2,1,0])
        data.shape = self.total_size,self.output_size,self.output_size,-1
        
        data_V = data1['VVVV']
        data_V = np.transpose(data_V,axes = [2,1,0])
        data_V = np.log10(data_V)*10
        data_V[data_V>0] = 0
        data_V[data_V<-25] = -25
        data_V = (data_V+25)/25
        data_V.shape = self.total_size,self.output_size,self.output_size
        
        if self.is_train == True:
            data_selected = np.zeros([self.training_size,self.output_size,self.output_size,9])
            data_selected[0:14,:,:,:]  = data[1:15,:,:,:]
            data_selected[14:18,:,:,:] = data[90:94,:,:,:]
            data_selected[18,:,:,:]    = data[78,:,:,:]
#            data_selected[19,:,:,:]   = data[105,:,:,:]
#            data_selected = np.reshape(data[90:94,:,:,:],[self.training_size,self.output_size,self.output_size,9])

            
            dataV_selected = np.zeros([self.training_size,self.output_size,self.output_size])
            dataV_selected[0:14,:,:]  = data_V[1:15,:,:]
            dataV_selected[14:18,:,:] = data_V[90:94,:,:]
            dataV_selected[18,:,:]    = data_V[78,:,:]
#            dataV_selected[19,:,:]   = data_V[105,:,:]
#            dataV_selected = np.reshape(data_V[90:94,:,:],[self.training_size,self.output_size,self.output_size])

            return data_selected,dataV_selected
        else:
#            matfn = './data/test_ScalBt_VVVV1.mat'
#            data1 = sio.loadmat(matfn)
#            test_V = data1['VVVV']
#            test_V = np.log10(test_V)*10
#            test_V[test_V>0] = 0
#            test_V[test_V<-40]  = -40
#            test_V = (test_V + 40)/40
            
#            matfn = './data/test_SF_VVVV.mat'
#            data1 = sio.loadmat(matfn)
#            test_V = data1['VVVV']
#            test_V = np.log10(test_V)*10
#            test_V[test_V>0] = 0
#            test_V[test_V<-25]  = -25
#            test_V = (test_V + 25)/25
            
            matfn = './data/test_MN150605_VVVV.mat'
            data1 = sio.loadmat(matfn)
            test_V = data1['VVVV']
            test_V = np.log10(test_V)*10
            test_V[test_V>0] = 0
            test_V[test_V<-25]  = -25
            test_V = (test_V + 25)/25
            
            
            matfn = './data/test_SD2_VVVV.mat'
            data1 = sio.loadmat(matfn)
            data_V = data1['VVVV']
            data_V[data_V==0] = 1
            data_V = np.log10(data_V)*10
            data_V[data_V>0] = 0
            data_V[data_V<-25] = -25
            data_V = (data_V+25)/25
            
            return data,data_V,test_V
 
        
            
        
        
def main(_):
    with tf.Session() as sess:
        sdgan = Pol_SD(sess)
    sdgan.train()
        
if __name__ == '__main__':
    tf.app.run() 
