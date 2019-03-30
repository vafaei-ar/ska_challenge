#####################################################################
# This code convert RA and DEC of radio sources from deg to pixel. #
# In addition it make cut and make images with desiered size around #
# each sources' center.												#
#####################################################################

from matplotlib import pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
import cv2
from astropy.nddata import Cutout2D


hdul = fits.open('C:/Users/mhaghighi/Downloads/SKAMid_B5_1000h_v3.fits')
train_set = np.loadtxt('C:/Users/mhaghighi/Downloads/TrainingSet_B5_v2.txt', skiprows=18)
hdu =hdul[0]
# Shape of SKAMid_B5_1000h_v3.fits is(1, 1, 32768, 32768) and  
# we need to reshape it 
image_datax= np.reshape(hdu.data,(32768,32768))
wcs = WCS(hdu.header).celestial


# In the previous versions of the training set we just had RA and DEC
# of point sources in deg. As a result by using the following wcs package
# we can convert cordinate from deg to pix.

ID = train_set[:,[0]]
size = (50, 50)     # pixels
x = train_set[:,1]
y = train_set[:,2]
cord = np.column_stack((x,y))
pixcrd = wcs.wcs_world2pix(cord, 1)
for i in range(5):
    cutout = Cutout2D(image_datax, pixcrd[i], size)
    fig = plt.figure()
    plt.imshow(cutout.data, origin='lower')
    #plt.savefig( str(ID[i][0]) + '.png')
    #plt.close()
	

################ This part is for visualization of the sources #######
######################################################################
	
# In orther to capture sources we have chosen two different pixel size
w1 = 10 ; h1 = 10
w2 = 150 ;h2 = 150
size1 = (w1, h1)     # pixels
size2 = (w2, h2)    
x = train_set[:,-2]
y = train_set[:,-1]
Lab_class = train_set[:,-4]

cord = np.column_stack((x,y))

for i in range(len(ID[:5])): #Just remove 5 to use all the training sets.
    cutout1 = Cutout2D(image_datax, cord[i], size1)
    cutout2 = Cutout2D(image_datax, cord[i], size2)
    
    fig = plt.figure()
    plt.ylabel('10*10 pixel size with class'+str(Lab_class[i] ))
    plt.imshow(cutout1.data, origin='lower')
    plt.savefig( str(ID[i][0]) + '-1.png')
  
    fig = plt.figure()
    plt.ylabel('150*150 pixel size with class'+str(Lab_class[i] ))
    plt.imshow(cutout2.data, origin='lower')
    
    plt.savefig( str(ID[i][0]) + '-2.png')
    plt.close()
	
	
################ This part provide the input of CNN ##################
######################################################################


data1 = np.zeros((train_set.shape[0],w1,h1,1),'uint8')
data2 = np.zeros((train_set.shape[0],w2,h2,1),'uint8')

label = np.uint8(train_set[:,-4])-1;
#feature = train_set[:,5:]
for i in range(10):
    cutout1 = Cutout2D(image_datax, cord[i], size1)
    cutout2 = Cutout2D(image_datax, cord[i], size2)

########################### normalize #########################
    fmin1 =  np.min(cutout1.data);fmax1 =  np.max(cutout1.data);
    temp1 = np.uint8(((cutout1.data - fmin1) / (fmax1-fmin1))*255)
    data1[i] = np.reshape(temp1,(w1,h1,1));
    
    fmin2 =  np.min(cutout2.data);fmax2 =  np.max(cutout2.data);
    temp2 = np.uint8(((cutout2.data - fmin2) / (fmax2-fmin2))*255)
    data2[i] = np.reshape(temp2,(w2,h2,1));
    
########################### save #########################
np.save('./DATA1_SKAMid_B1_8h_v2.npy',data1)    
np.save('./DATA2_SKAMid_B1_8h_v2.npy',data2)    
np.save('./LABEL_SKAMid_B1_8h_v2.npy',label)      
    

