# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 09:12:23 2018

@author: VHG6KOR
"""
import tensorflow as tf
from utils import preprocess, postprocess
from model import Col_Model
from data_loader_saver import data_loader_saver
import numpy as np
import matplotlib.pyplot as plt

class image_colorizer():
    
    def __init__(self):
        self.img_size=32
        self.epochs=50
        self.im_per_class=5


    def train(self):
        tf.reset_default_graph()
        col_model=Col_Model()
        dls=data_loader_saver()
        image_rgb = tf.placeholder(tf.float32, [None,self.img_size,self.img_size,3], name='image_rgb')
        # image_bw = tf.image.rgb_to_grayscale(image_rgb, name='image_bw')
        image_bw = tf.placeholder(tf.float32, [None,self.img_size,self.img_size,1], name='image_bw')
        image_col = preprocess(image_rgb)

        train=tf.placeholder(tf.bool, shape=(), name='train')
        global_step = tf.Variable(0.01, name='global_step', trainable=False)
        
        gen = col_model.generator(image_bw, train)
        dis_real = col_model.discriminator(tf.concat([image_bw, image_col], 3),train,False)
        dis_fake = col_model.discriminator(tf.concat([image_bw, gen], 3),train,True)
        
        smoothing=0.1
        rand1= tf.random_uniform(minval=0.7,maxval=1.2,dtype=tf.float32, shape=[1,4,4,1])
        rand2 = tf.random_uniform(minval=0,maxval=0.3,dtype=tf.float32, shape=[1,4,4,1])
        rand3 = tf.random_uniform(minval=0.7,maxval=1.2,dtype=tf.float32, shape=[1,4,4,1])

        gen_ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake, labels=rand3)
        dis_real_ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_real, labels=rand1 * smoothing)
        dis_fake_ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake, labels=rand2)

#        dis_loss_real = tf.reduce_mean(dis_real_ce)
#       dis_loss_fake = tf.reduce_mean(dis_fake_ce)
        d_loss = tf.reduce_mean(dis_real_ce + dis_fake_ce)
    
        gen_loss_gan = tf.reduce_mean(gen_ce)
        gen_loss_l1 = tf.reduce_mean(tf.abs(image_col- gen)) * 100
        g_loss = gen_loss_gan + gen_loss_l1
        
        t_vars = tf.trainable_variables()

        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]
        
        # learning_rate = tf.constant(3e-4)
        learning_rate = tf.maximum(1e-8, tf.train.exponential_decay(learning_rate=3e-4,   global_step=global_step, decay_steps=5e5, decay_rate=0.1))
        
        d_optim = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.0).minimize(d_loss, var_list=d_vars)
        g_optim = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.0).minimize(g_loss, var_list=g_vars)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver=tf.train.Saver()
            for planet in dls.planets:
                for idx in range(self.im_per_class):
                    imgs_col,imgs_bw =dls.load_train_batch(idx,planet)
                    for e in range(self.epochs):
                   
                        # Update D network
                        _,errD_real = sess.run([d_optim, d_loss],feed_dict={image_rgb:imgs_col,image_bw: imgs_bw, train:True })
        
                        _,errG= sess.run([g_optim,g_loss],feed_dict={ image_rgb:imgs_col,image_bw: imgs_bw,train:True })
                        print('dloss total:', errD_real,'g_loss:',errG, 'epoch:',e)
                    dls.save_model(sess,saver,planet,idx)
                    

      #  dls.pickle(image_bw,'image_bw') 
       # dls.pickle(image_rgb,'image_rgb') 
        #dls.pickle(gen,'gen') 
        #dls.pickle(train,'train') 
            
    def predict(self, image_bw_path, planet):
        dls=data_loader_saver()
        tf.get_variable_scope().reuse_variables()
#        tf.Graph
     #   image_bw = tf.get_variable("image_bw", shape=[1,self.img_size,self.img_size,1])
     #   train = tf.get_variable("train", shape=())
#        image_rgb = tf.get_variable("image_rgb", shape=[1,self.img_size,self.img_size,3])
     #   gen = tf.get_variable('gen', shape=[1,self.img_size,self.img_size,3])
        img_bw= dls.load_test_batch( image_bw_path)
        detection_graph = tf.Graph()
        with tf.Session(graph=detection_graph) as sess:
            sess.run(tf.global_variables_initializer()) 
            for planet in dls.planets:
                for idx in range(self.im_per_class):
                    dls.load_model(sess,planet,idx)
                    image_bw = detection_graph.get_tensor_by_name('image_bw:0')
                    g_loss= detection_graph.get_tensor_by_name('add_1:0')
                    train = detection_graph.get_tensor_by_name('train:0')
                    gen = detection_graph.get_tensor_by_name('gen:0')
                    image_rgb = detection_graph.get_tensor_by_name('image_rgb:0')
                    fake_image= sess.run([gen], feed_dict={image_bw: img_bw, train:False})
                    errG= sess.run([g_loss],feed_dict={ image_rgb:image_rgb,image_bw: img_bw,train:False })
                    fake_image=np.array(fake_image).reshape([1,self.img_size, self.img_size,3])
                    fake_image_post = postprocess(tf.convert_to_tensor(fake_image))
                    gen_image=fake_image_post.eval();
                    print('index: '+ str(idx)+'loss:'+str(errG))
                    #plt.imshow(gen_image[0])
                    #plt.show()
                    dls.save_gen_model(gen_image, planet, idx)
            
