from __future__ import division
import pickle
import numpy as np 
from collections import Counter 
from random import randint
import cv2




def get_index_label(ref):
    index1 = [] # Empty useful indices list
    label1 = [] # Empty labels list associated with useful indices
    index0 = [] # Empty zero value indices
    

    for i in range(991):     # number of rows in reference map
        arr = np.where(ref[i] !=0)    # check for useful indices
        arr0 = np.where(ref[i]==0)    # check for zero value indices
        if len(arr[0]) != 0:   
            for j in arr[0]:  
                index1.append([i,j])    # add all the useful indices in index1 list
                label1.append(ref[i][j])  # add associated labels to indices in label1 list
        if len(arr0[0]) != 0:
            for j in arr0[0]:
                index0.append([i,j])     # add all the zero value indices in index   
    return index0,index1, label1  # return zero_value indices list, useful indices list, real_value associated present in reference map

def generate_hundred_test(len_array,random_length):
    setOfNumbers = set()
    while len(setOfNumbers) < random_length:
        setOfNumbers.add(randint(0,len_array-1))
    random_array = list(setOfNumbers)    
    return np.array(random_array)

def best_solution( test_subject, class_num, iter_num, length_of_list):
	empty_list = [0]*8
	for i in range(iter_num):
		random_list = generate_hundred_test(len(test_subject),length_of_list)
		test_sub_try = test_subject[random_list]
		Counter_dict = Counter(test_sub_try)
		#print Counter_dict
		if i == 0:
			for counter_class in Counter_dict:
				empty_list[counter_class-1] = Counter_dict[counter_class]
		elif i > 0 and empty_list[class_num-1] < Counter_dict[class_num]:	
			empty_list = [0]*8
			for counter_class in Counter_dict:
				empty_list[counter_class-1] = Counter_dict[counter_class]
	list_order = np.array([6,7,4,5,1,2,3,0])
	#print "before",empty_list	
	empty_list = np.array(empty_list)[list_order]
	#print "after",empty_list
	return empty_list	

def get_pair_values(non_zero_values,logit_pic,real_ref):
	indexer = []
	data_dict ={}
	for index_num in non_zero_values:
		logit_value = logit_pic[index_num[0]][index_num[1]]
		real_value = real_ref[index_num[0]][index_num[1]]
		indexer.append([logit_value,real_value])
	total_num = len(indexer)
	for pair in indexer:
		if pair[0] in data_dict:
			data_dict[pair[0]].append(pair[1])
		else:
			data_dict[pair[0]] = [pair[1]]

	return data_dict,total_num		
def get_error_matrix(data_dict,total):
	length_list=[]
	error_matrix_sol = []

	for classes in data_dict:
		class_results = np.array(data_dict[classes])
		length = round((len(class_results)/total)*900)
		if length <= 50:
			error_matrix_sol.append(best_solution(class_results,classes, 1, 50))
			length_list.append(50)
		else:
			error_matrix_sol.append(best_solution(class_results,classes, 1, length))
			length_list.append(length)

	list_order = np.array([6,7,4,5,1,2,3,0])
	length_list = np.array(length_list)[list_order]
	error_matrix_sol = np.array(error_matrix_sol)[list_order]
	return error_matrix_sol,length_list

def calculate_acc_kappa(error_matrix_sol,length_list):
	column_sum = np.sum(error_matrix_sol, axis=0)
	diagonal_row = error_matrix_sol.diagonal()
	print "row_total", length_list, np.sum(length_list)
	print "column_total", column_sum, np.sum(column_sum)
	print "diagonal_total", diagonal_row, np.sum(diagonal_row)

	producer_accuracy = np.round(diagonal_row/column_sum,4)*100
	print "PA %", producer_accuracy 

	user_accuracy = np.round(diagonal_row/length_list,4)*100
	print "UA %", user_accuracy

	total_sample = np.sum(length_list)
	total_correct = np.sum(diagonal_row)
	conditional_kappa = np.round((total_sample*diagonal_row - length_list*column_sum)/(total_sample*length_list - length_list*column_sum),2)
	overall_kappa = (total_sample*total_correct - sum(length_list*column_sum))/(total_sample*total_sample - sum(length_list*column_sum))
	overall_acccuracy = total_correct/total_sample
	print "Conditional_Kappa:", conditional_kappa
	print "Overall Kappa:", "%.4f" %overall_kappa
	print "Overall Accuracy", "%.4f" %overall_acccuracy

	error_matrix_sol = np.hstack((error_matrix_sol,np.array([length_list]).T \
		,np.array([producer_accuracy]).T,np.array([user_accuracy]).T,np.array([conditional_kappa]).T))

	return error_matrix_sol


#pickle.dump(multi_prob, open("/home/atharva/Pictures/multi_prob_80.pkl", 'wb')) 
#logit_value_multi = np.argmax(multi_prob,axis = 2)+1
#multi_sp_80p_combined.tiff

logit_pic_use = cv2.imread("/home/atharva/Pictures/results_s2t/sp_rnn_80p.tiff",-1)
#logit_pic_use = cv2.imread("/home/atharva/Pictures/t5/dump80_pixel/149999test.tiff",-1)
#logit_pic = cv2.imread("/home/atharva/Pictures/fused_prob_pic80_t5.tiff",-1)
#logit_pic = cv2.imread("/home/atharva/Pictures/results_dict_year1_80.tiff",-1)
real_ref_use = cv2.imread("/home/atharva/Pictures/t5/used_ref8_new.tiff",-1)    

zero_index, non_zero , label = get_index_label(real_ref_use) # calling the function to get the indices associated with real value    
data_dict_class,total_count = get_pair_values(non_zero,logit_pic_use,real_ref_use)
error_matrix_old,length_list = get_error_matrix(data_dict_class,total_count)

print error_matrix_old
error_matrix_new = calculate_acc_kappa(error_matrix_old,length_list)
print error_matrix_new


