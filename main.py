import cv2
import numpy as np
import csv
import os
from datetime import datetime

import face_recognition

video_capture = cv2.VideoCapture(0)
 
tay_image =  face_recognition.load_image_file('faces/Taylor.train.jpg')
tay_encode = face_recognition.face_encodings(tay_image)[0]

known_face_encoding  = [tay_encode]
known_face = ["Taylor"]

celebrity =known_face.copy()

face_locations =  []
face_encodings = []
now = datetime.now()
current_date = now.strftime("%y-%m-%d")
s= True

f = open(f"{current_date}.csv" , "w+",newline="")
lnwrite = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame , (0,0) , fx = 0.25,fy = 0.25)
    rgb_small_frame  = cv2.cvtColor(small_frame , cv2.COLOR_BGR2RGB)

    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame)
        face_name=[]
       
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name=""
            face_distance = face_recognition.face_distance(known_face_encoding , face_encoding)
            best_match = np.argmin(face_distance)

            if(matches[best_match]):
                name = known_face[best_match]
            face_name.append(name)
            if name in known_face:
                font = cv2.FONT_HERSHEY_COMPLEX
                bottomleftcorner = (10 , 100)
                fontScale = 1.5
                fontColor = (255,0,0)
                thickness = 3
                linetype = 2
                cv2.putText(frame,name+"Present",bottomleftcorner ,font,fontScale,fontColor,thickness,linetype)
          

                if name in celebrity:
                    celebrity.remove(name)
                    print(celebrity)
                    current_time = now.strftime("%H-%M%S")  
                    lnwrite.writerow([name , current_time])  



    cv2.imshow("Attendance",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break        

video_capture.release()
cv2.destroyAllWindows()
f.close()    