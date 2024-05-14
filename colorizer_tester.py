# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 10:01:55 2018

"""

from image_colorizer import image_colorizer
from data_loader_saver import data_loader_saver
import os

ic=image_colorizer()
ic.train()

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path=dir_path+'\\Images\\Saturn\\BW\\Saturn_281.jpg'
dls=data_loader_saver()
ic.predict(dir_path,'Jupiter')
