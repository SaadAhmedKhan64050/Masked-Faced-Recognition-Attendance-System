import csv
import datetime
import encodings
import os
import time
from itertools import count

import cv2
import face_recognition
import imutils
import numpy as np
from imutils.video import VideoStream
# Get a reference to webcam #0 (the default one)
# import the necessary packages
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import load_model
from keras.utils import img_to_array

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.



def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds.append(maskNet.predict(faces, batch_size=32)[0].tolist())

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)


video_capture = cv2.VideoCapture(0)
img_path = os.getcwd() + "//Recog_Train"
images = []
known_face_names = []
known_face_encodings = []
encode_list_cl = []
myList = os.listdir(img_path)

#print(myList)

for subdir in os.listdir(img_path):
    path = img_path + '/' + subdir
    for subdir in os.listdir(path):
        path1 = path + '/' + subdir
        path1 = path1 + '/'
        for img in os.listdir(path1):
            img_pic = path1 + img
            known_face_names.append(subdir)
            cur_img = cv2.imread(img_pic)
            images.append(cur_img)
def find_encodings(images) :
    #for names in images :
        for img in images : 
            encodings = face_recognition.face_encodings(img)[0]
            known_face_encodings.append(encodings)
       
        return known_face_encodings
    
encodeListKnown = find_encodings(images)

# # Load a sample picture and learn how to recognize it.
# Saad_image = face_recognition.load_image_file("Saad.jpeg")
# Saad_face_encoding = face_recognition.face_encodings(Saad_image)[0]

# # Load a sample picture and learn how to recognize it.
# # Abdullah_image = face_recognition.load_image_file("Abdullah.jpeg")
# # Abdullah_face_encoding = face_recognition.face_encodings(Abdullah_image)[0]

# # Load a second sample picture and learn how to recognize it.
# Ayesha_image = face_recognition.load_image_file("Ayesha.jpeg")
# Ayesha_face_encoding = face_recognition.face_encodings(Ayesha_image)[0]

# Create arrays of known face encodings and their names
# known_face_encodings = [
#     Saad_face_encoding,
#     # Abdullah_face_encoding,
#      Ayesha_face_encoding
# ]
# known_face_names = [
#     "Saad",
#     # "Abdullah",
#      "Ayesha"
global tcount,count_
tcount=1
count_=1
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# initialize the video stream
print("[INFO] starting video stream...")
name1=""
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding)
            

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            name = "Unknown"
            tcount=tcount+1

            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            if(name==name1 and name!="Unknown"):
                count_=count_+1
            if(name!=name1):
            
                with open('.\Attendance.csv', 'a') as f:
                    date_time_string = datetime.datetime.now().strftime("%y/%m/%d %H:%M:%S")
                    f.writelines(f'\n{name},{date_time_string}')
            name1=name
            face_names.append(name)
            print(name)
            with open('.\Accuracy.csv', 'a') as f:
                    f.writelines(f'/n{name}')
            # csv_reader = csv.reader(f, delimiter=',')
            # count = 0
            # for row in csv_reader:        
            #     f.writelines(f'\n{name},')
            #     for row in f:
            #         count=+1
            #         print("total: ",count)
                        

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        if name == "Unknown":
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35),
                      (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name,(left + 6, bottom - 6),
                    font, 1.0, (255, 255, 255), 1)
    if(tcount>50):
        print((count_/tcount)*100)

        # show the output frame

    # Display the resulting image
    cv2.imshow('Video', frame)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
