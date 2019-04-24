'''
Running this code could take a while, especially done on you laptop.
We would recommend you copy the data from Google Drive to NYU HPC, 
change the path declared below and run this.

Inspired from the code given at: https://camelyon17.grand-challenge.org/Data/
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp
import glob

from PIL import Image
import openslide
from openslide.deepzoom import DeepZoomGenerator
from pathlib import Path
from scipy.misc import imsave as saveim
from skimage.filters import threshold_otsu

from pandas import HDFStore

import cv2 as cv2
from skimage import io

# If you need more information about how ElementTree package handle XML file, please follow the link: 
# https://docs.python.org/3/library/xml.etree.elementtree.html
import xml.etree.ElementTree as et
import pandas as pd
import math

# provide paths to the data files 
# NOTE: in our case, each folder only contained a couple of images
slide_path = '/avengers/harmanchawla/Downloads/CAMELYON16/training/tumor'
slide_path_normal = '/avengers/harmanchawla/Downloads/CAMELYON16/training/normal'

anno_path = '/avengers/harmanchawla/Downloads/CAMELYON16/training/Lesion_annotations'
BASE_TRUTH_DIR = '/avengers/harmanchawla/Downloads/CAMELYON16/masking'
slide_paths = glob.glob(osp.join(slide_path, '*.tif'))
slide_paths.sort()

slide_paths_normal = glob.glob(osp.join(slide_path_normal, '*.tif'))
slide_paths_normal.sort()

# slide_paths_total = slide_paths
slide_paths_total = slide_paths + slide_paths_normal


BASE_TRUTH_DIRS = glob.glob(osp.join(BASE_TRUTH_DIR, '*.tif'))
Anno_paths = glob.glob(osp.join(anno_path, '*.xml'))
BASE_TRUTH_DIRS.sort()


def convert_xml_df (file):
    '''
    
    The following parses an XML file and converts it into a dataframe. 
    The XML file contains a set of attributes over which we iterate and
    construct (store them) in the dataframe. 

    This dataframe is later used to generate a mask which is not
    provided to use by the organizers.

    @Args - accepts path to an XML file as input
    @Returns - dataframe coversion

    '''

    parseXML = et.parse(file)
    root = parseXML.getroot()
    dfcols = ['Name', 'Order', 'X', 'Y']
    new_XML = pd.DataFrame(columns=dfcols)

    for item in root.iter('Annotation'):
        for coordinate in item.iter('Coordinate'):
            Name = item.attrib.get('Name')
            Order = coordinate.attrib.get('Order')

            x_coordinate = float(coordinate.attrib.get('X'))
            Y_coord = float(coordinate.attrib.get('Y'))
           
            new_XML = new_XML.append(pd.Series([Name, Order, x_coordinate, Y_coord], index = dfcols), ignore_index=True)
            new_XML = pd.DataFrame(new_XML)


    return (new_XML)

# desired cropped size of each patch
crop_size = [256, 256]
i=0

while i < len(slide_paths):
   
    base_truth_dir = Path(BASE_TRUTH_DIR)
    anno_path = Path(anno_path)
    # checking if the name contains tumor. It is used later to do specific tasks.
    slide_contains_tumor = osp.basename(slide_paths_total[i]).startswith('tumor_')
    
    # for each slide for which we have been given a path
    with openslide.open_slide(slide_paths_total[i]) as slide:


        thumbnail = slide.get_thumbnail((slide.dimensions[0]/ 256, slide.dimensions[1]/ 256))
        thum = np.array(thumbnail)

        # just converting the color channel. Can also be done using sk Image.
        hsv_image = cv2.cvtColor(thum, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)

        # Otsu threshold perfomed on each channel
        hthresh = threshold_otsu(h)
        sthresh = threshold_otsu(s)
        vthresh = threshold_otsu(v)
        # be min value for v can be changed later
        minhsv = np.array([hthresh, sthresh, 70], np.uint8)
        maxhsv = np.array([180, 255, vthresh], np.uint8)
        thresh = [minhsv, maxhsv]
        #extraction the countor for tissue
        rgbbinary = cv2.inRange(hsv_image, thresh[0], thresh[1])

        #retriving the contours. We don't care about the other values. 
        _, contours, _ = cv2.findContours(rgbbinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initializing a DF bboxt which will be used to set the boundaries. 
        bboxtcols = ['xmin', 'xmax', 'ymin', 'ymax'] # bboxtcols underscores the structure of the df
        bboxt = pd.DataFrame(columns=bboxtcols)
        
        for c in contours:
            # returns the top-left co-ordinates along with width and height which are used to 
            # calculate the size and add it to the dataframe
            (x, y, w, h) = cv2.boundingRect(c)
            bboxt = bboxt.append(pd.Series([x, x+w, y, y+h], index = bboxtcols), ignore_index=True)
            bboxt = pd.DataFrame(bboxt)
         
        # each column of the dataframe is stored as a python list 
        # so that we can the max and min element in each column
        xxmin = list(bboxt['xmin'].get_values())
        xxmax = list(bboxt['xmax'].get_values())
        yymin = list(bboxt['ymin'].get_values())
        yymax = list(bboxt['ymax'].get_values())

        bboxt = math.floor(np.min(xxmin)*256), math.floor(np.max(xxmax)*256), math.floor(np.min(yymin)*256), math.floor(np.max(yymax)*256)
    
    # if it is a file with tumor in it's name
    if slide_contains_tumor:

        # get the paths to the mask and annotations 
        truth_slide_path = base_truth_dir / osp.basename(slide_paths_total[i]).replace('.tif', '_mask.tif')
        Anno_pathxml = anno_path / osp.basename(slide_paths_total[i]).replace('.tif', '.xml')

        with openslide.open_slide(str(truth_slide_path)) as truth:

          slide = openslide.open_slide(slide_paths_total[i])

          # convert xml to a temp dataframe
          annotations = convert_xml_df(str(Anno_pathxml))

          # just as above, get the co-ordinate values and calculate the boundary values 
          x_values = list(annotations['X'].get_values())
          y_values = list(annotations['Y'].get_values())
          bbox = math.floor(np.min(x_values)), math.floor(np.max(x_values)), math.floor(np.min(y_values)), math.floor(np.max(y_values))
          
          temp=0
         
          while temp in range(0, 1000):
            # r = [rgb_image, rgb_binary, rgb_mask, index]
            r=random_crop(slide, truth, thresh, crop_size, bbox)
            if (cv2.countNonZero(r[2]) > crop_size[0]*crop_size[1]*0.5) and (temp <= 1000):
    
                saveim('/avengers/harmanchawla/Downloads/test/tumor/%s_%d_%d.png' % (osp.splitext(osp.basename(slide_paths_total[i]))[0], r[3][0], r[3][1]), r[0])
                io.imsave('/avengers/harmanchawla/Downloads/test/mask/%s_%d_%d_mask.png' % (osp.splitext(osp.basename(slide_paths_total[i]))[0], r[3][0], r[3][1]), r[2])
                print(r[2])
                temp = temp +1
            

    else:
        temp=0    
        slide = openslide.open_slide(slide_paths_total[i])

        while temp in range(0, 1000):
            # nr = [gb_image, rgb_binary, index]
            nr=random_crop_normal(slide, thresh, crop_size, bboxt)
            
            if (cv2.countNonZero(r[1]) > crop_size[0]*crop_size[1]*0.2) and (temp <= 1000):
               nmask = np.zeros((256, 256))

               saveim('/avengers/harmanchawla/Downloads/test/normal/%s_%d_%d.png' % (osp.splitext(osp.basename(slide_paths_total[i]))[0], nr[2][0], nr[2][1]),nr[0])
               io.imsave('/avengers/harmanchawla/Downloads/test/nmask/%s_%d_%d_mask.png' % (osp.splitext(osp.basename(slide_paths_total[i]))[0], nr[2][0], nr[2][1]), nmask)
               temp= temp+1

    i=i+1

# cropping tumor images 
def random_crop(slide, truth, thresh, crop_size, bbox):
    
    # it might help to recall what boundary box (bbox) contains:
    # min(x_values), max(x_values), min(y_values)), max(y_values)
    # crop size = [256, 256]
    dy, dx = crop_size
    x = np.random.randint(bbox[0], bbox[1] - dx + 1)
    y = np.random.randint(bbox[2], bbox[3] - dy + 1)
    
    index=[x, y]
    #print(index)
    
    rgb_image = slide.read_region((x, y), 0, crop_size)
    rgb_mask = truth.read_region((x, y), 0, crop_size)

    rgb_mask = (cv2.cvtColor(np.array(rgb_mask), cv2.COLOR_RGB2GRAY) > 0).astype(int) # convert RGB
    rgb_array = np.array(rgb_image) # store as NP array
    hsv_rgbimage = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2HSV) # convert HSV
    rgb_binary = cv2.inRange(hsv_rgbimage, thresh[0], thresh[1]) # apply thresholds
    # print(index)

    return (rgb_image, rgb_binary, rgb_mask, index)

# cropping normal images (Pretty much same as above, but We don't need a mask.)
def random_crop_normal(slide, thresh, crop_size, bboxt):
    
    dy, dx = crop_size
    x = np.random.randint(bboxt[0], bboxt[1] - dx + 1)
    y = np.random.randint(bboxt[2], bboxt[3] - dy + 1)
    index=[x, y]
    
    rgb_image = slide.read_region((x, y), 0, crop_size)
  
    rgb_array = np.array(rgb_image)
    hsv_rgbimage = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2HSV)
    rgb_binary = cv2.inRange(hsv_rgbimage, thresh[0], thresh[1])
   
    return (rgb_image, rgb_binary, index)


def testduplicates(list):   
    for each in list:  
        count = list.count(each)  
        if count > 1:  
            z = 0
        else:
        
            z = 1
    return z  