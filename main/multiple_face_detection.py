import face_recognition
import cv2
import numpy as np
import os
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import wget
import dlib
import time
import csv
 
recordAttendance = []
known_face_encodings = []
known_face_names = []
 
path = "Pictures"
for folder in os.listdir(path):
    for image in os.listdir("{}/{}".format(path, folder)):
        face_image = face_recognition.load_image_file("{}/{}/{}".format(path, folder, image))
        Face_encoding = face_recognition.face_encodings(face_image)[0]
        known_face_encodings.append(Face_encoding)
        known_face_names.append("{}".format(folder))
 
video_capture = cv2.VideoCapture(0)
 
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
 
while True:
    ret, frame = video_capture.read()
    name = "Unknown"
    if process_this_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
 
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        prevName = 'Default'
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
 
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                if name != "Unknown" and prevName:
                    now = time.ctime()
                    recordAttendance.append("{}, {}".format(name, now))
 
            face_names.append(name)
            if len(recordAttendance) < 2:
                print(recordAttendance)
 
        process_this_frame = not process_this_frame
 
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale the face locations back up
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
 
        # Draw the face rectangle and name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
 
    process_this_frame = not process_this_frame
 
 
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
 
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
 
    cv2.imshow('Video', frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
video_capture.release()
cv2.destroyAllWindows()