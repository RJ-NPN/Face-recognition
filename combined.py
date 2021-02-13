
import cv2 
import numpy as np 
import npwriter  
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from collections import Counter

from npwriter import f_name 

import os 
from sklearn.metrics import accuracy_score
  
from sklearn.model_selection import cross_val_score


x=input("enter your choice:")
print(type(x))
if(x=='1'):
    name = input("Enter your name: ") 
    cap = cv2.VideoCapture(0)   
    classifier = cv2.CascadeClassifier("C:\\Users\\Admin\\Desktop\\bin\\py\\zip\\opencv-master\\data\\lbpcascades\\lbpcascade_frontalface_improved.xml") 
    f_list = [] 
      
    while True: 
        ret, frame = cap.read() 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)       
        faces = classifier.detectMultiScale(gray, 1.5, 5) 
        faces = sorted(faces, key = lambda x: x[2]*x[3], 
                                         reverse = True)   
        faces = faces[:1]   
        if len(faces) == 1:    
            face = faces[0]    
            

            x, y, w, h = face  
         
            im_face = frame[y:y + h, x:x + w]  
            cv2.imshow("face", im_face)    
        if not ret: 
            continue  
        cv2.imshow("full", frame)   
        key = cv2.waitKey(1)  
        if key & 0xFF == ord('q'): 
            break
        elif key & 0xFF == ord('c'): 
            if len(faces) == 1: 
                gray_face = cv2.cvtColor(im_face, cv2.COLOR_BGR2GRAY) 
                gray_face = cv2.resize(gray_face, (500, 500)) 
                print(len(f_list), type(gray_face), gray_face.shape) 
      
                f_list.append(gray_face.reshape(-1))  
            else: 
                print("face not found") 
      
  
    npwriter.write(name, np.array(f_list))
    cap.release() 
    cv2.destroyAllWindows() 

if(x=='2'):
    data = pd.read_csv(f_name).values   
    X, Y = data[:, 1:-1], data[:, -1]   
    print(X, Y)
    
    model1 = KNeighborsClassifier(n_neighbors = 5)   
    model1.fit(X, Y)

    model2 = svm.SVC()
    model2.fit(X,Y)

    model3 = RandomForestClassifier(n_estimators=10)
    model3.fit(X, Y)

    model4 = GaussianNB()
    model4.fit(X,Y)

    model5 = LogisticRegression(random_state=0)
    model5.fit(X,Y)
 
    
    cap = cv2.VideoCapture(0)
    i=5
    #cap = "F:\\major\\chadda\\"+"("+str(i)+")"+".png"
    classifier = cv2.CascadeClassifier("C:\\Users\\Admin\\Desktop\\bin\\py\\zip\\opencv-master\\data\\lbpcascades\\lbpcascade_frontalface_improved.xml")   
    f_list = []   
    while True:  
        ret,frame = cap.read()   
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
        faces = classifier.detectMultiScale(gray, 1.5, 5)   
        X_test = []
        
        for face in faces: 
            x, y, w, h = face 
            im_face = gray[y:y + h, x:x + w] 
            im_face = cv2.resize(im_face, (500, 500)) 
            X_test.append(im_face.reshape(-1)) 
    
        
        if len(faces)>0: 
            a = model1.predict(np.array(X_test))

            b = model2.predict(np.array(X_test))

            c = model3.predict(np.array(X_test))

            d = model4.predict(np.array(X_test))

            e = model5.predict(np.array(X_test))

            print("       prediction over         ")

            


            List = [str(a),str(b),str(c),str(d),str(e)]
            count = Counter(List)
            v=count.most_common(1)[0][0]
            print(v)

            for i, face in enumerate(faces): 
                x, y, w, h = face   
                cv2.rectangle(frame, (x, y), (x + w, y + h), 
                                             (255, 0, 0), 3)   
                cv2.putText(frame, str(v), (10, 50), 
                                  cv2.FONT_HERSHEY_DUPLEX, 1, 
                                             (0, 255, 0), 2)
        cv2.imshow("full", frame)   
        key = cv2.waitKey(1)   
        if key & 0xFF == ord("q") : 
            break  
    cap.release() 
    cv2.destroyAllWindows() 

    
