# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 23:46:16 2018

@author: VHG6KOR
"""
from PIL import Image
import numpy as np
import os 
import pickle
import tensorflow as tf

class data_loader_saver():
    
    
    def __init__(self):
        self.planets=['Jupiter','Saturn','Mars','Callisto']
        self.planets=['Jupiter','Saturn']
        self.planets=['Saturn']
        self.img_size=32
        self.images_path='Images'
    
    def load_train_batch(self,idx, planet):
        
        dir_path = os.path.dirname(os.path.realpath(__file__))
        planet_color_path=dir_path+'/'+self.images_path+'/'+planet+'/'+'Color'+'/'+planet+'_'+str(idx)+'.jpg'
        img_col = np.array(Image.open(planet_color_path))
        img_col=img_col.reshape([1,self.img_size, self.img_size, 3])
        
        img_bw = np.array(Image.open(planet_color_path).convert('L'))
        img_bw=img_bw.reshape([1,self.img_size, self.img_size, 1])     

        return img_col/255, img_bw/255  

    def load_test_batch(self, images_bw_path):
        img_bw = np.array(Image.open(images_bw_path).convert('L'))
        img_bw=img_bw.reshape([1,self.img_size, self.img_size, 1])     

        return  img_bw/255

    def save_model(self,sess, saver, planet, idx):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        planet_model_path=dir_path+'/'+self.images_path+'/'+planet+'/'+'Model'+'/'+str(idx)+'/'+planet+'_'+str(idx)+'.jpg'
        saver.save(sess, planet_model_path)
        
    def load_model(self,sess, planet, idx):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        planet_metamodel_path=dir_path+'/'+self.images_path+'/'+planet+'/'+'Model'+'/'+str(idx)+'/'+planet+'_'+str(idx)+'.jpg'+'.meta'
        datapath = dir_path+'/'+self.images_path+'/'+planet+'/'+'Model'+'/'+str(idx)+'/'+planet+'_'+str(idx)+'.jpg'+'.data-00000-of-00001'
        ckptdirpath = dir_path+'/'+self.images_path+'/'+planet+'/'+'Model'+'/'+str(idx)
        print('planet_metamodel_path',planet_metamodel_path)
        print('datapath',datapath)
        saver = tf.train.import_meta_graph(planet_metamodel_path)
        saver.restore(sess, tf.train.latest_checkpoint(ckptdirpath))
        return saver
        
    def save_gen_model(self, gen_image, planet, idx):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        planet_model_path=dir_path+'/'+self.images_path+'/'+planet+'/'+'Gen'+'/'+planet+'_'+str(idx)+'.jpg'
        im1 = Image.fromarray((gen_image[0]*255).astype(np.uint8))
        im1.save(planet_model_path)
        
    def pickle(self, obj, objname):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        pickle_path=dir_path+'/'+'Pickle'+'/'+objname+'.pkl'
        with open(pickle_path, 'wb') as f:
            pickle.dump(obj,f)
            
    def unpickle(self, objname):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        pickle_path=dir_path+'/'+'Pickle'+'/'+objname+'.pkl'
        with open(pickle_path, 'rb') as f:
            obj = pickle.load(f)
        return obj
        
        
    
        