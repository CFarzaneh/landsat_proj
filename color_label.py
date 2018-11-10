import numpy as np 
import cv2
import pickle
path = "/home/atharva/Pictures/results_s2t/"
#ref=cv2.imread('zero_pic_conv8.tiff',-1)
ref = cv2.imread(path + "used_ref8_new.tiff",-1)

red_c =  np.zeros(shape=(991,864))
green_c = np.zeros(shape=(991,864))
blue_c = np.zeros(shape=(991,864))
def create_direc(list_class):
	int_dir = {}
	for i in list_class:
		int_dir[i] = []
	return int_dir
def get_index(value):
	index1 = []
	global ref
	for i in range(991):
		arr = np.where(ref[i] == value)
		if len(arr[0]) != 0:
			for j in arr[0]:
				index1.append([i,j])
	return index1

def unique_array_class(class_name,label_dict,color_dict):
	for index in label_dict[class_name]:
		red_c[index[0]][index[1]] = color_dict[class_name][0] 
		green_c[index[0]][index[1]] = color_dict[class_name][1]
		blue_c[index[0]][index[1]] = color_dict[class_name][2]

list1 = np.unique(ref)
intial_dir = create_direc(list1)


intial_dir[1] =[0,0,255]  #Blue : Water
intial_dir[2] =[0,255,0]  #Lime  :Agriculture
intial_dir[3] =[255,0,255] #Pink/Magenta : Forested Wetland
intial_dir[4] = [128,0,128] #Purple: No forested
intial_dir[5] = [0,255,255]  #Aqua/Cyan  : Barren Land
intial_dir[6] = [0,128,0]  #Green : Forest
intial_dir[7] = [255,0,0] #Red : High Intensity
intial_dir[8] = [128,0,0] #Maroon : Low Intensity
intial_dir[0] = [0,0,0]
#intial_dir[100] = [0,0,0]

'''
intial_dir[1] =[0,0,255]  #Blue : Water
intial_dir[2] =[0,255,0]  #Lime  :Agriculture
intial_dir[3] =[255,0,255] #Pink/Magenta : Wetland
intial_dir[4] = [128,0,128] #Purple: Barren
intial_dir[5] = [0,128,0]  #Green : Forest
intial_dir[6] = [255,0,0] #Red : High Intensity
intial_dir[0] = [0,0,0]
'''


for i in intial_dir:
	if intial_dir[i] == []:
		intial_dir[i] = [0,0,0]

dict1 = {}	

for element in list1:
	# storing indices of all the classes in dict1 dictionary
	dict1[element] = get_index(element)	

#pickle.dump(dict1, open('lab6conv_use.pkl', 'wb'))

for i in list1:
	unique_array_class(i,dict1,intial_dir)

#np.save('redb.npy',red_c)
#np.save('greenb.npy',green_c)
#np.save('blueb.npy',blue_c)		
img = cv2.merge((blue_c,green_c,red_c))
#cv2.imwrite("conv8_ref_map.jpg", img)
cv2.imwrite(path + "original_ref_map.jpg", img)