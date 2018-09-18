
# coding: utf-8

# In[ ]:

import numpy as np 
import cv2 
import pickle
import copy
import math
from random import randint
import os
path = "/Users/cfarzaneh/Desktop/LC08_L1TP_016042_20140210_20170307_01_T1/"
sample_path = "/Users/cfarzaneh/Desktop/8class_new60percent/"

#num_class_list = [6]
#pix_dim_list = [5,7]
num_class = 8
pix_dim = 9
#ref = cv2.imread(path + "ref8.tiff",-1)
output = np.load(path + "outfileLC08_L1TP_016042_20140210_20170307_01_T1.npy")

def ready_and_execute(num_class,pix_dim):
    
    ref = cv2.imread(path + "ref8_new.tiff",-1)
    print(ref.shape)
    unique_ref = np.unique(ref)
    list1 = np.delete(unique_ref,[0,len(unique_ref)-1])

    return ref, list1
    #return ref, unique_ref


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

def cloud_level_count_up(ind_num,ref,num_id,level_num,max_level):
    count = 0
    for i in range(max_level+1):
        if i == 0:
            if np.any(ref[ind_num[0]-level_num][ind_num[1]]) == 0:
                count = count + 1
        else:
            if np.any(ref[ind_num[0]-level_num][ind_num[1]-i]) == 0:
                count = count + 1
            if np.any(ref[ind_num[0]-level_num][ind_num[1]+i]) == 0:
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
             

def cloud_level_count_down(ind_num,ref,num_id,level_num,max_level):
    count = 0
    for i in range(max_level+1):
        if i == 0:
            if np.any(ref[ind_num[0]+level_num][ind_num[1]]) == 0:
                count = count + 1
        else:
            if np.any(ref[ind_num[0]+level_num][ind_num[1]-i]) == 0:
                count = count + 1
            if np.any(ref[ind_num[0]+level_num][ind_num[1]+i]) == 0:
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

def cloud_level_count_side(ind_num,ref,num_id,max_level):
    count = 0
    for i in range(1,max_level+1):
        if np.any(ref[ind_num[0]][ind_num[1]-i]) == 0:
            count = count + 1
        if np.any(ref[ind_num[0]][ind_num[1]+i]) == 0:
            count = count + 1           
            
    return count  


def neighbour_count(ind_num, ref,num_id, max_level):
    temp_count = 0
    for i in range(1,max_level+1):
        l_count = level_count_up(ind_num,ref,num_id,i,max_level) + level_count_down(ind_num,ref,num_id,i,max_level)
        temp_count = temp_count + l_count
    cover = temp_count +  level_count_side(ind_num,ref,num_id,max_level) + 1 
    return cover 

def cloud_neighbour_count(ind_num, ref,num_id, max_level):
    temp_count = 0
    for i in range(1,max_level+1):
        l_count = cloud_level_count_up(ind_num,ref,num_id,i,max_level) + cloud_level_count_down(ind_num,ref,num_id,i,max_level)
        temp_count = temp_count + l_count
    cover = temp_count +  cloud_level_count_side(ind_num,ref,num_id,max_level) 
    return cover 

def avoid_boundary(ind_num,rows, columns, max_level):
    touch_boundary = 0
    for i in range(max_level):
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
    temp = []
    temp_pixel = []
    rows = len(ref)-1
    columns = len(ref[0])-1
    sixty_percent_value = math.floor(pow(image_dim,2)*0.6)
    for ind_num in index_file:
        
        if avoid_boundary(ind_num,rows, columns, max_level) == 0:
            if np.any(out_image[ind_num[0]][ind_num[1]]) != 0:
                cover = neighbour_count(ind_num,ref,num_id,max_level)
                clouded_cover = cloud_neighbour_count(ind_num, out_image,num_id, max_level)
   
                if cover > sixty_percent_value and clouded_cover == 0:
                    ul_y = ind_num[0]-max_level
                    ul_x = ind_num[1]-max_level
                    pixel_sample = out_image[ind_num[0]][ind_num[1]]
                    sample = out_image[ul_y:ul_y+ image_dim, ul_x:ul_x+ image_dim ]
                    temp_pixel.append(pixel_sample)
                    temp.append(sample)

            
            
    return np.array(temp_pixel),np.array(temp)           
            

def generate_labels(num_list):
    labels = []
    min_value = -1
    for classes,num in zip(list1,num_list):
        labels.extend([classes]*num)
    if len(num_list) != 0:
        min_value = min(num_list)
    
    return np.array(labels), min_value
def permutate_samples_order(samples_pixel,samples_data,labels_data):
    if len(samples_data) == len(labels_data) and len(samples_data) == len(samples_pixel):
        permutated_list = np.random.permutation(len(samples_data))
        X_pixel_data = samples_pixel[permutated_list]
        X_patch_data = samples_data[permutated_list]
        Y_data = labels_data[permutated_list]
        return X_pixel_data,X_patch_data,Y_data,permutated_list
    else:
        return None,None,None,None
    
def create_test_label(label,num_class):
    new_label = []
    for i in range(100):
        temp = [0]*num_class
        temp[label-1]=1
        new_label.append(temp)
    return np.array(new_label,np.float32)

    #Y_train_new = create_label(Y_train)
def create_label(list_lab,num_class):
    new_label = []
    for i in list_lab:
        temp = [0]*num_class
        temp[i-1]=1
        new_label.append(temp)
    return np.array(new_label,np.float32)

def generate_hundred_test(len_array):
    setOfNumbers = set()
    while len(setOfNumbers) < 100:
        setOfNumbers.add(randint(0,len_array-1))
    random_array = list(setOfNumbers)    
    return np.array(random_array)

def split_samples(pixel_samples,samples):
	samples_dict = {}
	num_samples = len(samples)
	test_index = generate_hundred_test(num_samples)
	test_pixel_samples = pixel_samples[test_index]
	test_samples =samples[test_index]
	print ("test_samples", len(test_samples), len(test_pixel_samples))
	train_pixel_samples = np.delete(pixel_samples,test_index,0)
	training_samples = np.delete(samples,test_index,0)
	samples_dict ["patch"] = [test_samples,training_samples]
	samples_dict ["pixel"] = [test_pixel_samples,train_pixel_samples]
	#print "training_sample", len(training_samples)
	return samples_dict 

def hundred_percent(num_class,conv_dim):
    folder_conv = sample_path +str(num_class)+'classes_with_path_dim_'+str(conv_dim)
    folder_pixel = sample_path +str(num_class)+'classes_single_pix_sameas_with_path_dim_'+str(conv_dim)
    if not os.path.exists(folder_conv):
        os.makedirs(folder_conv)
    if not os.path.exists(folder_pixel):
        os.makedirs(folder_pixel) 
           

    temp_data = []
    temp_pix_data = []
    temp_number = []
    for classes in list1:
        dataset ={}
        print ("total samples of", classes , "is",len(dict1[classes]))
        temp = dict1[classes]
        pixel_data,conv_data = cut_conv_image(temp, output, ref,int(classes),conv_dim)
        print ("Before split ",len(conv_data), len(pixel_data))
        if len(conv_data) > 2000:
            dataset = split_samples(pixel_data,conv_data)
            label = int(classes)
            test_labels = create_test_label(label,num_class)
            np.save(folder_conv+'/'+ 'test_patch_of_'+str(label) +'.npy',dataset["patch"][0])
            np.save(folder_pixel+'/'+ 'test_pixel_of_'+str(label) +'.npy',dataset["pixel"][0])
            np.save(folder_conv+'/'+ 'test_label_'+str(label) +'.npy',test_labels)
            np.save(folder_pixel+'/'+ 'test_label_'+str(label) +'.npy',test_labels)
            #test_samples, train_samples = split_samples(conv_data)'''
              
        if "patch" in dataset:
            if len(dataset["patch"][1]) != 0:
                print ("After split",len(dataset["patch"][1]),len(dataset["pixel"][1]))
                temp_data.extend(dataset["patch"][1])
                temp_pix_data.extend(dataset["pixel"][1])
                temp_number.append(len(dataset["patch"][1]))
                dataset = {}
           
    temp_data = np.array(temp_data)
    temp_pix_data = np.array(temp_pix_data)
    temp_labels, temp_min = generate_labels(temp_number)
    print ("num",len(temp_data), len(temp_pix_data))
    print ("temp_number", len(temp_labels))
    print ("min_value",temp_min)
    X_pixel_train, X_patch_train, Y_train, permut_list = permutate_samples_order(temp_pix_data,temp_data,temp_labels)
    Y_train_new = create_label(Y_train,num_class)
    if X_patch_train is not None:
        np.save(folder_conv+'/'+ 'X_train_patch.npy',X_patch_train)
        np.save(folder_conv+'/'+'Y_train_patch.npy',Y_train_new)
        print ("++++++++++++++")
    else:
        print ("Unexpected")

    if X_pixel_train is not None:
        np.save(folder_pixel+'/'+ 'X_train_pixel.npy',X_pixel_train)
        np.save(folder_pixel+'/'+'Y_train_pixel.npy',Y_train_new)
        print ("++++++++++++++")
    else:
        print ("Unexpected")

'''for num_class in num_class_list:
    for pix_dim in pix_dim_list:
        print "Sample with:", num_class ,"classes and a patch dimension :", pix_dim
        print "------------------------------------------------------------------"
        ref, list1 = ready_and_execute(num_class,pix_dim)

        dict1 = {}     
        for element in list1:
                   # storing indices of all the classes in dict1 dictionary
            dict1[element] = get_index(element,ref)


        hundred_percent(num_class,pix_dim)
        #ref,list1,dict1 = None
'''


print ("Sample with:", num_class ,"classes and a patch dimension :", pix_dim)
print ("------------------------------------------------------------------")
ref, list1 = ready_and_execute(num_class,pix_dim)

dict1 = {}     
for element in list1:
           # storing indices of all the classes in dict1 dictionary
    dict1[element] = get_index(element,ref)

hundred_percent(num_class,pix_dim)