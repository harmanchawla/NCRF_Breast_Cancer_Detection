import csv
import glob
import os
import random

import cv2
import numpy as np
import scipy.stats.stats as st
from skimage.measure import label
from skimage.measure import regionprops

FILTER_DIM = 2
N_FEATURES = 31
MAX, MEAN, VARIANCE, SKEWNESS, KURTOSIS = 0, 1, 2, 3, 4

def format_2f(number):
    return float("{0:.2f}".format(number))

def get_region_props(heatmap_threshold_2d, heatmap_prob_2d):
    labeled_img = label(heatmap_threshold_2d)
    return regionprops(labeled_img, intensity_image=heatmap_prob_2d)


def get_largest_tumor_index(region_props):
    largest_tumor_index = -1
    
    largest_tumor_area = -1
    
    n_regions = len(region_props)
    for index in range(n_regions):
        if region_props[index]['area'] > largest_tumor_area:
            largest_tumor_area = region_props[index]['area']
            largest_tumor_index = index

    return largest_tumor_index


def get_longest_axis_in_largest_tumor_region(region_props, largest_tumor_region_index):
    largest_tumor_region = region_props[largest_tumor_region_index]
    return max(largest_tumor_region['major_axis_length'], largest_tumor_region['minor_axis_length'])


def get_feature(region_props, n_region, feature_name):
    feature = [0] * 5
    if n_region > 0:
        feature_values = [region[feature_name] for region in region_props]
        feature[MAX] = format_2f(np.max(feature_values))
        feature[MEAN] = format_2f(np.mean(feature_values))
        feature[VARIANCE] = format_2f(np.var(feature_values))
        feature[SKEWNESS] = format_2f(st.skew(np.array(feature_values)))
        feature[KURTOSIS] = format_2f(st.kurtosis(np.array(feature_values)))
    
    return feature


def get_average_prediction_across_tumor_regions(region_props):
    # close 255
    region_mean_intensity = [region.mean_intensity for region in region_props]
    return np.mean(region_mean_intensity)


def extract_features(heatmap_prob):
    """
        Feature list:
        -> (01) given t = 0.90, total number of tumor regions
        -> (02) given t = 0.50, the area of largest tumor region
        -> (03) given t = 0.50, the longest axis in the largest tumor region
        -> (04) given t = 0.90, total number pixels with probability greater than 0.90
        -> (05) given t = 0.90, average prediction across tumor region
        -> (06-10) given t = 0.90, max, mean, variance, skewness, and kurtosis of 'area'
        -> (11-15) given t = 0.90, max, mean, variance, skewness, and kurtosis of 'perimeter'
        -> (16-20) given t = 0.90, max, mean, variance, skewness, and kurtosis of 'eccentricity'
        -> (21-25) given t = 0.50, max, mean, variance, skewness, and kurtosis of 'rectangularity(extent)'
        -> (26-30) given t = 0.90, max, mean, variance, skewness, and kurtosis of 'solidity'
        
        :param heatmap_prob:
        :return:
        
        """
    
    heatmap_threshold_t90 = np.array(heatmap_prob)
    heatmap_threshold_t50 = np.array(heatmap_prob)
    heatmap_threshold_t90[heatmap_threshold_t90 < 0.90] = 0
    heatmap_threshold_t90[heatmap_threshold_t90 >= 0.90] = 255
    heatmap_threshold_t50[heatmap_threshold_t50 <= 0.50] = 0
    heatmap_threshold_t50[heatmap_threshold_t50 > 0.50] = 255
    
    heatmap_threshold_t90_2d = heatmap_threshold_t90
    heatmap_threshold_t50_2d = heatmap_threshold_t50
    heatmap_prob_2d = heatmap_prob

    region_props_t90 = get_region_props(np.array(heatmap_threshold_t90_2d), heatmap_prob_2d)
    region_props_t50 = get_region_props(np.array(heatmap_threshold_t50_2d), heatmap_prob_2d)
    
    features = []
                                          
    f_count_tumor_region = len(region_props_t90)

    features.append(format_2f(f_count_tumor_region))

    largest_tumor_region_index_t90 = get_largest_tumor_index(region_props_t90)
    largest_tumor_region_index_t50 = get_largest_tumor_index(region_props_t50)
    f_area_largest_tumor_region_t50 = region_props_t50[largest_tumor_region_index_t50].area
    features.append(format_2f(f_area_largest_tumor_region_t50))

    f_longest_axis_largest_tumor_region_t50 = get_longest_axis_in_largest_tumor_region(region_props_t50, largest_tumor_region_index_t50)
    features.append(format_2f(f_longest_axis_largest_tumor_region_t50))

    f_pixels_count_prob_gt_90 = cv2.countNonZero(heatmap_threshold_t90_2d)
    features.append(format_2f(f_pixels_count_prob_gt_90))

    if f_count_tumor_region is not 0:
        f_avg_prediction_across_tumor_regions = get_average_prediction_across_tumor_regions(region_props_t90)
    else:
        f_avg_prediction_across_tumor_regions = 0
    features.append(format_2f(f_avg_prediction_across_tumor_regions))

    f_area = get_feature(region_props_t90, f_count_tumor_region, 'area')
    features += f_area
    
    f_perimeter = get_feature(region_props_t90, f_count_tumor_region, 'perimeter')
    features += f_perimeter
    
    f_eccentricity = get_feature(region_props_t90, f_count_tumor_region, 'eccentricity')
    features += f_eccentricity
    
    f_extent_t50 = get_feature(region_props_t50, len(region_props_t50), 'extent')
    features += f_extent_t50
    
    f_solidity = get_feature(region_props_t90, f_count_tumor_region, 'solidity')
    features += f_solidity
    
    return features

if __name__ == '__main__':
    toy_sample = ['patient_000_node_3', #0, negative
                  'patient_005_node_3', #1, itc
                  'patient_007_node_4', #2, micro
                  'patient_013_node_3', #3, macro
                  'patient_020_node_1', #4, micro
                  'patient_020_node_3', #5, negative
                  'patient_020_node_4', #6, macro
                  'patient_021_node_2', #7, negative
                  'patient_100_node_0', #8, negative, test
                  'patient_105_node_3', #9, itc, test
                  'patient_107_node_4', #10, micro, test
                  'patient_112_node_0' #11, macro, test
                  ]
    filename = toy_sample[0]
    probs_map = np.load('probs_map/' + filename + '.npy')
    features = extract_features(probs_map)
    
    
    for filename in toy_sample:
        outfile = open('features/' + filename + '.csv', 'w')
        for feature in features:
            outfile.write('{:0.5f}'.format(feature) + '\n')
        outfile.close()
