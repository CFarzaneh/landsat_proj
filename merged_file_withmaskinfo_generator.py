import numpy as np 
import cv2
import math as m
import glob

dir_path="/Users/cfarzaneh/Desktop/"

def get_value(i,line):
	if i+' =' in line:
		return float((line.replace(i+' =', "")))

def get_givenfile_values(date,value1,value2,value3,value4,value5): 
	a=None
	b=None
	c=None
	d=None
	e=None	
	global dir_path
	
	path=dir_path+ date+"/*.txt"
	for file1 in glob.glob(path):
	    with open(file1,'r') as searchfile:
	    	for line in searchfile:
	    		    			
	    		if a == None:
	    			a= get_value(value1,line)
	    		if b == None:
	    			b= get_value(value2,line)
	    		if c == None:
	    			c= get_value(value3,line)
	    		if d == None:
	    			d= get_value(value4,line)
	    		if e == None:
	    			e= get_value(value5,line)		
	    		if a!= None and b != None and c !=None and d !=None and e !=None:
	    			# get UL_X_Value and UL_Y_Value in respect to the reference label image
	    			v,w,x,y,z = a,b,c,int((412080-d)/30),int((e-2933460)/30)
	    			print (v,w,x,y,z)
	    			a=None
	    			b=None
	    			c=None
	    			d=None
	    			e=None
	    			return v,w,x,y,z	


def reflectance(reflectance_mult,reflectance_add,sun_elevation_angle,raw_value):
	new_value=(raw_value*reflectance_mult + reflectance_add)/m.sin(m.radians(sun_elevation_angle))
	return new_value
			

def get_useful_image(doi):
	global dir_path
	ref_mult,ref_add,sun_elev_angle,X1_p,Y1_p = get_givenfile_values(doi,\
		'REFLECTANCE_MULT_BAND_4','REFLECTANCE_ADD_BAND_4',\
		'SUN_ELEVATION','CORNER_UL_PROJECTION_X_PRODUCT','CORNER_UL_PROJECTION_Y_PRODUCT')
	
	raw1 = cv2.imread(str(glob.glob(dir_path + doi +'/*_B1.TIF')[0]),-1)[Y1_p:Y1_p+991,X1_p:X1_p+864]
	raw2 = cv2.imread(str(glob.glob(dir_path + doi +'/*_B2.TIF')[0]),-1)[Y1_p:Y1_p+991,X1_p:X1_p+864]
	raw3 = cv2.imread(str(glob.glob(dir_path + doi +'/*_B3.TIF')[0]),-1)[Y1_p:Y1_p+991,X1_p:X1_p+864]
	raw4 = cv2.imread(str(glob.glob(dir_path + doi +'/*_B4.TIF')[0]),-1)[Y1_p:Y1_p+991,X1_p:X1_p+864]
	raw5 = cv2.imread(str(glob.glob(dir_path + doi +'/*_B5.TIF')[0]),-1)[Y1_p:Y1_p+991,X1_p:X1_p+864]
	raw6 = cv2.imread(str(glob.glob(dir_path + doi +'/*_B6.TIF')[0]),-1)[Y1_p:Y1_p+991,X1_p:X1_p+864]
	raw7 = cv2.imread(str(glob.glob(dir_path + doi +'/*_B7.TIF')[0]),-1)[Y1_p:Y1_p+991,X1_p:X1_p+864]
	raw9 = cv2.imread(str(glob.glob(dir_path + doi +'/*_B9.TIF')[0]),-1)[Y1_p:Y1_p+991,X1_p:X1_p+864]
	
	mask_part = cv2.imread(str(glob.glob(dir_path + doi +'/*_Mask.tif')[0]),-1)[Y1_p:Y1_p+991,X1_p:X1_p+864]
	mask_part = np.repeat(mask_part[:, :, np.newaxis], 8, axis=2)
	
	new1 = reflectance(ref_mult,ref_add,sun_elev_angle,raw1)
	new2 = reflectance(ref_mult,ref_add,sun_elev_angle,raw2)
	new3 = reflectance(ref_mult,ref_add,sun_elev_angle,raw3)
	new4 = reflectance(ref_mult,ref_add,sun_elev_angle,raw4)
	new5 = reflectance(ref_mult,ref_add,sun_elev_angle,raw5)
	new6 = reflectance(ref_mult,ref_add,sun_elev_angle,raw6)
	new7 = reflectance(ref_mult,ref_add,sun_elev_angle,raw7)
	new9 = reflectance(ref_mult,ref_add,sun_elev_angle,raw9)

	merge_image = np.dstack((new1,new2,new3,new4,new5,new6,new7,new9))
	
	final_image = merge_image * mask_part
	
	np.save(dir_path + doi +'/outfile'+ doi +'.npy', final_image)
	

date_of_interest = ["LC08_L1TP_016042_20140210_20170307_01_T1"]

#date_of_interest1 = ["2014_12_27","2015_01_12","2015_01_28"]

for d_interest in date_of_interest:
	get_useful_image(d_interest)

	

	
