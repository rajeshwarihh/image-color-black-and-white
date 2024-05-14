# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 22:33:14 2018
"""
import tensorflow as tf

class Col_Model():
    
    def __init__(self):
        self.seed=100
        self.kernel_size=4
    
    def generator(self, z, train):
        with tf.variable_scope("generator"):
            n=str(1)
            s='g'
            conv1= tf.layers.conv2d(inputs=z,kernel_size=self.kernel_size, strides=1,kernel_initializer=tf.variance_scaling_initializer(seed=self.seed), filters=64, padding='same', name=s+'_conv'+n)
            bias1 = tf.get_variable(s+'_bias'+n, [64], initializer=tf.constant_initializer(0.0))
            conv1=tf.nn.bias_add(conv1, bias1)
	#         b1= tf.contrib.layers.batch_norm(conv, decay=0.9, updates_collections=None, epsilon=1e-5,center=True, scale=True, is_training=train, scope=s+'_b'+n)
            r1= self.lrelu(conv1)
			
	#         conv1 = conv2d_g(z,64, train, 1,1) #32
            conv2 = self.conv2d_g(r1,128, train, 2)  #16
            conv3 = self.conv2d_g(conv2,256, train, 3)  #8
            conv4 = self.conv2d_g(conv3,512, train, 4)  #4
            conv5 = self.conv2d_g(conv4,512, train, 5)#2
	
            deconv1 = self.transpose_conv2d(conv5, 512, train, 6) #4
            drop1= tf.layers.dropout(deconv1, rate=0.5, name='g_drop1', training=train)
            cc1 = self.copy_concat(conv4,drop1) 
			
            deconv2 = self.transpose_conv2d(cc1, 256, train, 7) #8
            drop2= tf.layers.dropout(deconv2, rate=0.5, name='g_drop2', training=train)
            cc2 = self.copy_concat(conv3, drop2)
			
            deconv4 = self.transpose_conv2d(cc2, 128, train,8) #16
            cc3 = self.copy_concat(conv2,deconv4)
			
            deconv5 = self.transpose_conv2d(cc3, 64, train, 9) #32
            cc4 = self.copy_concat(r1,deconv5)
			
			
            n='10'
            conv= tf.layers.conv2d(inputs=cc4,kernel_size=1, strides=1,kernel_initializer=tf.variance_scaling_initializer(seed=self.seed), filters=3, padding='same', name='g_conv'+n)
            bias = tf.get_variable('g_bias'+n, [3], initializer=tf.variance_scaling_initializer(seed=self.seed))
            conv=tf.nn.bias_add(conv, bias)
	#         b= tf.contrib.layers.batch_norm(conv, decay=0.9, updates_collections=None, epsilon=1e-5,center=True, scale=True, is_training=train, scope='g_b'+n)
	
        return tf.nn.tanh(conv, name='gen')
    
    def discriminator(self,image, train, reuse):
        with tf.variable_scope("discriminator"):
            if reuse:    
                tf.get_variable_scope().reuse_variables()
            s='d'
            n=str(1)
            conv1= tf.layers.conv2d(inputs=image,kernel_size=self.kernel_size, strides=2,kernel_initializer=tf.variance_scaling_initializer(seed=self.seed), filters=64, padding='same', name=s+'_conv'+n)
            bias1 = tf.get_variable(s+'_bias'+n, [64], initializer=tf.constant_initializer(0.0))
            conv1=tf.nn.bias_add(conv1, bias1)
            r1 = self.lrelu(conv1)
	#         conv1 = conv2d_d(image, fs=64, i=1, strides=2, train=train, ks=kernel_size)#16
            conv2 = self.conv2d_d(r1, fs=128, i=2, strides=2, train=train, ks=self.kernel_size)#8
            conv3 = self.conv2d_d(conv2, fs=256, i=3, strides=2, train=train, ks=self.kernel_size)#4
			#4
            conv4 = self.conv2d_d(conv3, fs=512, i=4, strides=1, train=train, ks=self.kernel_size)
			
            n=str(5)
			
            conv5= tf.layers.conv2d(inputs=conv4,kernel_size=self.kernel_size, strides=1,kernel_initializer=tf.variance_scaling_initializer(seed=self.seed), filters=1, padding='same', name=s+'_conv'+n)
            bias5 = tf.get_variable(s+'_bias'+n, [1], initializer=tf.constant_initializer(0.0))
            conv5=tf.nn.bias_add(conv5, bias5)
			
            return conv5
    
    def copy_concat(self,x,y):
        cx= tf.identity(x)
        return tf.concat(axis=3, values=[cx,y])
    
    def conv2d_d(self, x, fs, train, i, ks, strides):
        n=str(i)
        s='d'
        conv= tf.layers.conv2d(inputs=x,kernel_size=ks, strides=strides,kernel_initializer=tf.variance_scaling_initializer(seed=self.seed), filters=fs, padding='same', name=s+'_conv'+n)
        bias = tf.get_variable(s+'_bias'+n, [fs], initializer=tf.constant_initializer(0.0))
        conv=tf.nn.bias_add(conv, bias)
        b= tf.contrib.layers.batch_norm(conv, decay=0.9, updates_collections=None, epsilon=1e-5,center=True, scale=True, is_training=train, scope=s+'_b'+n)
        return self.lrelu(b)
    
    def conv2d_g(self, x, fs, train, i, strides=2):
        n=str(i)
        s='g'
        conv= tf.layers.conv2d(inputs=x,kernel_size=self.kernel_size, strides=strides,kernel_initializer=tf.variance_scaling_initializer(seed=self.seed), filters=fs, padding='same', name=s+'_conv'+n)
        bias = tf.get_variable(s+'_bias'+n, [fs], initializer=tf.constant_initializer(0.0))
        conv=tf.nn.bias_add(conv, bias)
        b= tf.contrib.layers.batch_norm(conv, decay=0.9, updates_collections=None, epsilon=1e-5,center=True, scale=True, is_training=train, scope=s+'_b'+n)
        return self.lrelu(b)
    
    def transpose_conv2d(self, x, fs, train, i):
        n=str(i)
        deconv= tf.layers.conv2d_transpose(inputs=x, kernel_initializer=tf.variance_scaling_initializer(seed=self.seed), filters=fs, kernel_size=self.kernel_size, name='g_deconv'+n, padding='same', strides=2)
        bias = tf.get_variable('g_bias'+n, [fs], initializer=tf.constant_initializer(0.0))
        deconv=tf.nn.bias_add(deconv, bias)
        b= tf.contrib.layers.batch_norm(deconv, decay=0.9, updates_collections=None, epsilon=1e-5,center=True, scale=True, is_training=train, scope='g_b'+n)
        return tf.nn.relu(b)
                
    def lrelu(self, x, leak=0.2, name="lrelu"):
        return tf.maximum(leak * x, x)
