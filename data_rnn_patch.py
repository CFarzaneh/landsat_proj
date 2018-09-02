
# coding: utf-8

# In[ ]:

import numpy as np 
import cv2 
import pickle
import copy
import math
from random import randint
import os
path = "/home/atharva/Pictures/outfile/"

date_of_interest = ["2014_02_10","2014_02_26","2014_03_14","2014_03_30",\
"2014_04_15","2014_05_01","2014_05_17","2014_06_02","2014_06_18","2014_07_04","2014_07_20",\
"2014_08_05","2014_08_21","2014_09_06","2014_09_22","2014_10_08","2014_10_24","2014_11_09",\
"2014_11_25","2014_12_11","2014_12_27","2015_01_12","2015_01_28"]
#sample_path = "/home/atharva/Pictures/t24/9_samples/8class_new60percent/"


num_class = 8
pix_dim = 3




def get_index(value,ref):
    index1 = []
    #global ref
    for i in range(991):
        arr = np.where(ref[i] == value)
        if len(arr[0]) != 0:
            for j in arr[0]:
                index1.append([i,j])
    return index1

def level_count_up(ind_num,ref,num_id,level_num,max_level):
    count = 0
    for i in range(max_level+1):
        if i == 0:
            if ref[ind_num[0]-level_num][ind_num[1]] == num_id:
                count = count + 1
        else:
            if ref[ind_num[0]-level_num][ind_num[1]-i] == num_id:
                count = count + 1
            if ref[ind_num[0]-level_num][ind_num[1]+i] == num_id:
                count = count + 1 
                
    return count




def level_count_down(ind_num,ref,num_id,level_num,max_level):
    count = 0
    for i in range(max_level+1):
        if i == 0:
            if ref[ind_num[0]+level_num][ind_num[1]] == num_id:
                count = count + 1
        else:
            if ref[ind_num[0]+level_num][ind_num[1]-i] == num_id:
                count = count + 1
            if ref[ind_num[0]+level_num][ind_num[1]+i] == num_id:
                count = count + 1  
              
    return count 
             

def level_count_side(ind_num,ref,num_id,max_level):
    count = 0
    for i in range(1,max_level+1):
        if ref[ind_num[0]][ind_num[1]-i] == num_id:
            count = count + 1
        if ref[ind_num[0]][ind_num[1]+i] == num_id:
            count = count + 1           
            
    return count  







def neighbour_count(ind_num, ref,num_id, max_level):
    temp_count = 0
    for i in range(1,max_level+1):
        l_count = level_count_up(ind_num,ref,num_id,i,max_level) + level_count_down(ind_num,ref,num_id,i,max_level)
        temp_count = temp_count + l_count
    cover = temp_count +  level_count_side(ind_num,ref,num_id,max_level) + 1 
    return cover 



def avoid_boundary(ind_num,rows, columns):
    touch_boundary = 0
    for i in range(2):
        if ind_num[0] - i == 0:
            touch_boundary = touch_boundary +1
        if ind_num[0] + i == rows:
            touch_boundary = touch_boundary +1
        if ind_num[1] - i == 0:
            touch_boundary = touch_boundary +1
        if ind_num[1] + i == columns:
            touch_boundary = touch_boundary +1  
    return touch_boundary       
    
                
            
    
    

def cut_conv_image(index_file,out_image, ref,num_id, image_dim):
    max_level = image_dim//2
    count = 0
    temp_train = []
    temp_possible = []
    #temp_pixel = []
    rows = len(ref)-1
    columns = len(ref[0])-1
    sixty_percent_value = math.floor(pow(image_dim,2)*0.6)
    for ind_num in index_file:
        
        if avoid_boundary(ind_num,rows, columns) == 0:
            cover = neighbour_count(ind_num,ref,num_id,max_level)
            ul_y = ind_num[0]-max_level
            ul_x = ind_num[1]-max_level
            sample = out_image[ul_y:ul_y+ image_dim, ul_x:ul_x+ image_dim ]
            sample = sample.flatten()
            temp_possible.append(sample)
            if cover == 9 : 
                temp_train.append(sample)

        
    return np.array(temp_train), np.array(temp_possible)           
            

def generate_labels(num_list):
    labels = []
    min_value = -1
    for classes,num in zip(list1,num_list):
        labels.extend([classes]*num)
    if len(num_list) != 0:
        min_value = min(num_list)
    
    return np.array(labels), min_value
def permutate_samples_order(samples_pixel,samples_data,labels_data):
    if len(samples_data) == len(labels_data) :
        permutated_list = np.random.permutation(len(samples_data))
        X_patch_data = samples_data[permutated_list]
        Y_data = labels_data[permutated_list]
        return X_patch_data,Y_data,permutated_list
    else:
        return None,None,None

def create_label(list_lab,num_class):
    new_label = []
    for i in list_lab:
        temp = [0]*num_class
        temp[i-1]=1
        new_label.append(temp)
    return np.array(new_label,np.float32)

    



def hundred_percent(num_class,conv_dim,output_file):

    train_full_data = []
    possible_full_data = []
    temp_possible_number = []
    temp_train_number = []
    for classes in list1:
        print "total samples of", classes , "is",len(dict1[classes])
        temp = dict1[classes]
        conv_data, possible_conv_data = cut_conv_image(temp, output_file, ref,int(classes),conv_dim)
        train_full_data.extend(conv_data)
        temp_train_number.append(len(conv_data))
        possible_full_data.extend(possible_conv_data)
        temp_possible_number.append(len(possible_conv_data))
           
    train_full_data = np.array(train_full_data)
    train_labels, train_min = generate_labels(temp_train_number)
    possible_full_data = np.array(possible_full_data)
    possible_labels, possible_min = generate_labels(temp_possible_number)

    return train_full_data, temp_train_number, possible_full_data, temp_possible_number

    



print "Sample with:", num_class ,"classes and a patch dimension :", pix_dim
print "------------------------------------------------------------------"
ref = cv2.imread("/home/atharva/Pictures/t5/used_ref8_new.tiff",-1)
list1 = range(1,num_class+1)

dict1 = {}     
for element in list1:
           # storing indices of all the classes in dict1 dictionary
    dict1[element] = get_index(element,ref)

main_data_file_train = []
main_label_file_train = []
main_data_file_possible = []
main_label_file_possible = []
for date in date_of_interest:
    output_num = np.load(path + "outfile"+date+".npy")
    m_d_f_t, m_l_f_t, m_d_f_p, m_l_f_p = hundred_percent(num_class,pix_dim,output_num)
    main_data_file_train.append(m_d_f_t)
    main_label_file_train.append(m_l_f_t)
    main_data_file_possible.append(m_d_f_p)
    main_label_file_possible.append(m_l_f_p)
    
    
main_data_file_train = np.array(main_data_file_train)
main_label_file_train = np.array(main_label_file_train)
main_data_file_possible = np.array(main_data_file_possible)
main_label_file_possible = np.array(main_label_file_possible)


main_data_file_train = np.column_stack(main_data_file_train)
#main_label_file_train = np.array(main_label_file_train)
main_data_file_possible = np.column_stack(main_data_file_possible)
#main_label_file_possible = np.array(main_label_file_possible)
 
print "Training Samples"
print "num ",len(main_data_file_train)
print "train_number", len(main_label_file_train[0])
#print "min_value_train",train_min

print "Possible Samples"
print "num ",len(main_data_file_possible)
print "train_number", len(main_label_file_possible[0])
#print "min_value_train",possible_min
    #X_patch_train, Y_train, permut_list = permutate_samples_order(temp_pix_data,temp_data,temp_labels)
    #Y_train_new = create_label(Y_train,num_class)

np.save(path + 'X_train_spatial_temp.npy',main_data_file_train)
np.save(path + 'Y_train_spatial_temp.npy',main_label_file_train[0])
np.save(path + 'check_train_Y.npy',main_label_file_train)
print "++++++++++++++"


np.save(path + 'X_possible_spatial_temp.npy',main_data_file_possible)
np.save(path + 'Y_possible_spatial_temp.npy',main_label_file_possible[0])
np.save(path + 'check_possible_Y.npy',main_label_file_possible)
print "++++++++++++++"
   