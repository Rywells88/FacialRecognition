#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
from matplotlib import pyplot as plt
import numpy as np
import time


# In[2]:


def cascade(img_L,img_C,img_R):
    
    face_cascade = cv2.CascadeClassifier('cascade.xml')


    gray_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
    gray_C = cv2.cvtColor(img_C, cv2.COLOR_BGR2GRAY)
    gray_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)


    faces_L = face_cascade.detectMultiScale(gray_L, 1.3, 5)
    faces_C = face_cascade.detectMultiScale(gray_C, 1.3, 5)
    faces_R = face_cascade.detectMultiScale(gray_R, 1.3, 5)

    largest = []
    position = []

    for (x,y,w,h) in faces_L:
        largest.append(w+h)
        position.append(((x,y),(w,h)))
    if(len(largest) != 0):
        biggestBox = max(largest)
        biggestBoxIndex = largest.index(max(largest))

        (x,y) = position[biggestBoxIndex][0]
        (w,h) = position[biggestBoxIndex][1]

        x1,y1 = x,y
        x2,y2 = x+w,y+h

        img_L_cropped = img_L[y1+2:(y2-1), x1+2:(x2-1)]
        img_L = cv2.rectangle(img_L,(x,y),(x+w,y+h),(255,0,0),2)
    else:
        # --- no boxes were found  -----
        img_L_cropped = img_L

    

    largest.clear()
    position.clear()
    for (x,y,w,h) in faces_C:
        largest.append(w+h)
        position.append(((x,y),(w,h)))
    if(len(largest) != 0):
        biggestBox = max(largest)
        biggestBoxIndex = largest.index(max(largest))

        (x,y) = position[biggestBoxIndex][0]
        (w,h) = position[biggestBoxIndex][1]

        x1,y1 = x,y
        x2,y2 = x+w,y+h

        img_C_cropped = img_C[y1+2:(y2-1), x1+2:(x2-1)]
        img_C = cv2.rectangle(img_C,(x,y),(x+w,y+h),(255,0,0),2)
    else:
         # --- no boxes were found  -----
        img_C_cropped = img_C

    


    largest.clear()
    position.clear()
    for (x,y,w,h) in faces_R:
            largest.append(w+h)
            position.append(((x,y),(w,h)))
    if(len(largest) != 0):
        biggestBox = max(largest)
        biggestBoxIndex = largest.index(max(largest))

        (x,y) = position[biggestBoxIndex][0]
        (w,h) = position[biggestBoxIndex][1]

        x1,y1 = x,y
        x2,y2 = x+w,y+h

        img_R_cropped = img_R[y1+2:(y2-1), x1+2:(x2-1)]
        img_R = cv2.rectangle(img_R,(x,y),(x+w,y+h),(255,0,0),1)
    else:
        # --- no boxes were found  -----
        img_R_cropped = img_R

    # return cropped images containing what's inside each box 
    return (img_L_cropped,img_C_cropped,img_R_cropped)


# In[3]:


def confusionMatrix(myDict):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for key,value in myDict.items():
        if("real" in key and value == "Real"): # real image classified as real : TP
            TP += 1
        if("real" in key and value == "Fake"): # real image classified as fake : FN
            FN += 1
        if ("fake" in key and value == "Fake"):# fake image classified as fake : TN
            TN += 1
        if("fake" in key and value == "Real"): # fake image classified as real : FP
            FP += 1
            
    print("TP: ", TP)
    print("FN: ", FN)
    print("TN: ", TN)
    print("FP: ", FP)
    
    print("Precision : ", TP/(TP + FP))
    print("Recall : ", TP/(TP+FN))


# In[4]:


# 1.1 Matching
def SIFT_Lowes(image1, image2):



    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1,None)
    
    kp2, des2 = sift.detectAndCompute(image2,None)
    
    
    
    bf = cv2.BFMatcher()
    matches1 = bf.knnMatch(des1,des2, k=2)
   

    
    
    matchesMask1 = [[0,0] for i in range(len(matches1))]
    
    
    matches = []

    Lowes_matches = []

    
    for i,(m,n) in enumerate(matches1): # backpack
        matches.append(m.distance)
        if m.distance < 0.7*n.distance:
            
            Lowes_matches.append(m)
            matchesMask1[i]=[1,0]
                
    return ((kp1,kp2),(des1,des2), Lowes_matches)


# In[5]:


# Epipolar lines
def findHomography(left, right):
    
    sizeleft = left.shape[0] + left.shape[1]
    sizeright = right.shape[0] + right.shape[1]
    
    if(sizeleft>sizeright):
        left = cv2.resize(left,(right.shape[0],right.shape[1]))
    elif(sizeleft < sizeright):
        right = cv2.resize(right,(left.shape[0],left.shape[1]))

    result = SIFT_Lowes(left, right)

    kp1 = result[0][0]
    kp2 = result[0][1]
    
    des1 = result[1][0]
    des2 = result[1][1]
    
    matches = result[2]
    
    ref_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    img_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    
    
    if(len(ref_pts) < 4 or len(img_pts) < 4):
        return False
    
    M, mask = cv2.findHomography(ref_pts, img_pts, cv2.RANSAC, 5.0)
    
    if(M is None):
        return False
    else:
        return True


# In[6]:


def classify_homography():
    
    start_time = time.time()
    RF = ["real","fake"]
    
    results = {}
    
    # ----- To swap out dataset: 
    # j represents the folders of real and fake throughout the loop. - adjust this in for loop if it's different than 12 
    # i represents how many sets of images there are. (There's 12 sets in data provided)
    # L,C,R - left, centre, right - can change to anything 
    # j[0] - just takes the first letter of either real or fake 
    #
    
    
    for j in RF:
        for i in range(1,12):
   # -----  Example path for a picture : real/10Cr.jpg
        
            img_L = cv2.imread(j + '/'+ str(i) + 'L' + j[0] + '.jpg')
            img_C = cv2.imread(j + '/'+ str(i) + 'C' + j[0] + '.jpg')
            img_R = cv2.imread(j + '/'+ str(i) + 'R' + j[0] + '.jpg')
            

            cascade_result = cascade(img_L, img_C, img_R)


            numPoints_LR = findHomography(cascade_result[0], cascade_result[2])
            numPoints_RC = findHomography(cascade_result[2], cascade_result[1])
            numPoints_LC = findHomography(cascade_result[0], cascade_result[1])
            
            
   
            numPoints = numPoints_LR and numPoints_RC and numPoints_LC
  
            if(numPoints == True):

                results[j + '/' + str(i) + ' '  + '.jpg'] = "Fake"
            else:

                results[j + '/' + str(i) + ' '  + '.jpg'] = "Real"

        
    print(results)
    print(len(results))
    confusionMatrix(results)
    print ("Elapsed time in seconds ", time.time() - start_time)
    
classify_homography()


# In[ ]:




