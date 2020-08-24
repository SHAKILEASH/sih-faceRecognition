
import cv2 
import dlib
import numpy as np
import PIL.Image

import glob

def whirldata_face_detectors(img, number_of_times_to_upsample=1):
 return face_detector(img, number_of_times_to_upsample)

def whirldata_face_encodings(face_image,num_jitters=1):
 face_locations = whirldata_face_detectors(face_image)
 pose_predictor = face_pose_predictor
 predictors = [pose_predictor(face_image, face_location) for face_location in face_locations]
 return [np.array(face_encoder.compute_face_descriptor(face_image, predictor, num_jitters)) for predictor in predictors][0]

def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)

def get_coords(p1):
    try: return int(p1[0][0][0]), int(p1[0][0][1])
    except: return int(p1[0][0]), int(p1[0][1])

face_detector = dlib.get_frontal_face_detector()
predictor_model = "shape_predictor_68_face_landmarks.dat"
face_pose_predictor = dlib.shape_predictor(predictor_model)
face_encoder = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
max_head_movement = 20
movement_threshold = 50
gesture_threshold = 175
face_found = False
frame_num = 0
cap = cv2.VideoCapture(0) 

while frame_num < 30:
    # Take first frame and find corners in it
    frame_num += 1
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(frame_gray, 1)
    #print(faces)
    face_rect=faces[0]
   # print(face_rect)
    (x,y,w,h) = (face_rect.left(),face_rect.right(),face_rect.top(),face_rect.bottom())
    face_found = True
    cv2.imshow('Head Gesture Detection',frame)
    cv2.waitKey(1)
face_center = y/2, w/3
p0 = np.array([[face_center]], np.float32)
print(p0)



found_names = []
targets_name = []
targets=[]
txtfiles = [] 
for file in glob.glob("faces\*.jpg"):
    buff=file[6:-4]
    targets_name.append(buff)
    txtfiles.append(file)
print(targets_name)
for ix in txtfiles:
    img = cv2.imread(ix,cv2.IMREAD_COLOR)
    enc = whirldata_face_encodings(img)
    targets.append(enc)




gesture = False
x_movement = 0
y_movement = 0
gesture_show = 60 #number of frames a gesture is shown
frame_count = 0
tolerance=0.6
while 1:  
    ret, img = cap.read() 
    detected_faces = face_detector(img, 1)
    #print(detected_faces)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
    found_names = []
    for i, face_rect in enumerate(detected_faces):
        x=face_rect.left()
        y = face_rect.right()
        top = face_rect.top()
        bottom = face_rect.bottom()
        cv2.rectangle(img,(x,top),(y,bottom),(255,255,0),2) 
        pose_landmarks = face_pose_predictor(img, face_rect)
        encod=[np.array(face_encoder.compute_face_descriptor(img, pose_landmarks, 1))][0]

        toleffect=list(face_distance(targets, encod) <= tolerance)
        if len(encod) == 0:
         dist=np.empty((0))
        else:		 
            dist= np.linalg.norm(targets - encod, axis=1)
        fit=np.argmin(dist)
        if (toleffect[fit] == True):
            found_names.append(targets_name[fit])            
        print(found_names)
  
    cv2.imshow('img',img) 
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break
  

cap.release() 

cv2.destroyAllWindows()