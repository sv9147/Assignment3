import cv2
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
from statistics import mean
from skimage import data
from skimage.filters import threshold_multiotsu
from skimage.feature import corner_harris


class Preprocess:

    def histogram_equlization_rgb(self, img):
        # Simple preprocessing using histogram equalization 
        # https://en.wikipedia.org/wiki/Histogram_equalization

        intensity_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        intensity_img[:, :, 0] = cv2.equalizeHist(intensity_img[:, :, 0])
        img = cv2.cvtColor(intensity_img, cv2.COLOR_YCrCb2BGR)

        # For Grayscale this would be enough:
        # img = cv2.equalizeHist(img)

        return img

    # Add your own preprocessing techniques here.
    
    def contrastAdjustment(self, img):
    
        b = 30
        c = 70
    
        new_img = np.zeros(img.shape, img.dtype)
        
        HSL = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        H, S, L_contrast = cv2.split(HSL)

        limit = 255 - c
        
        L_contrast[L_contrast > limit] = 255
        L_contrast[L_contrast <= limit] += c
        
        corrected_hsl = cv2.merge((H, S, L_contrast))
        new_img = cv2.cvtColor(corrected_hsl, cv2.COLOR_HLS2BGR)
        
        #cv2.imwrite(filename + '.contrast.jpg', new_img)
        
        return new_img
        
    def brightnessCorrection(self, img):
    
        b = 70
        c = 50
        
        #values that slightly improve face detection:  b = 50 and c = 40
        #values that improve ear detection: b = 30 and c = 20  
        
        new_img = np.zeros(img.shape, img.dtype)
        
        HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H, S, V_brightness = cv2.split(HSV)

        limit = 255 - b
       
        #popravimo vrednosti, glede na podani value
        V_brightness[V_brightness > limit] = 0
        V_brightness[V_brightness <= limit] += b

        

        corrected_hsv = cv2.merge((H, S, V_brightness))
        new_img = cv2.cvtColor(corrected_hsv, cv2.COLOR_HSV2BGR)
        
        #cv2.imwrite(filename + '.brightness.jpg', new_img)
        
      
        return new_img
    
    def sharpenImage(self, img):
    
    	new_img = np.zeros(img.shape, img.dtype)
    	kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    	new_img = cv2.filter2D(img, -1, kernel)
    	
		
    	return new_img
    	
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    	
