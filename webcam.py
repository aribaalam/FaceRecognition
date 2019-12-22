Python 3.7.0 (v3.7.0:1bf9cc5093, Jun 26 2018, 23:26:24) 
[Clang 6.0 (clang-600.0.57)] on darwin
Type "copyright", "credits" or "license()" for more information.
>>> import cv2
import time
from pretrain import *
import os.path


def detect_face(database, model):
    save_loc = r'saved_image/1.jpg'
    capture_obj = cv2.VideoCapture(0)
    capture_obj.set(3, 640)  
    capture_obj.set(4, 480)  


   
    req_sec = 3
    loop_start = time.time()
    elapsed = 0

    while(True):
        curr_time = time.time()
        elapsed = curr_time - loop_start
        if elapsed >= req_sec:
            break
        ret, frame = capture_obj.read()
        frame = cv2.flip(frame, 1, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi_color = frame[y-90:y+h+70, x-50:x+w+50]
            cv2.imwrite(save_loc, roi_color)
            cv2.rectangle(frame, (x-10, y-70),
                          (x+w+20, y+h+40), (15, 175, 61), 4)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture_obj.release()
    cv2.destroyAllWindows()

    img = cv2.imread(save_loc)
    if img is not None:
        face_found = True
    else:
        face_found = False

    return face_found
    




def detect_face_realtime(database, model, threshold=0.7):
    text = ''
    font = cv2.FONT_HERSHEY_SIMPLEX
    save_loc = r'saved_image/1.jpg'
    capture_obj = cv2.VideoCapture(0)
    capture_obj.set(3, 640)  
    capture_obj.set(4, 480)  

    face_cascade = cv2.CascadeClassifier(
        r'haarcascades/haarcascade_frontalface_default.xml')
    print('**************** Enter "q" to quit **********************')
    prev_time = time.time()
    while(True):

        
        ret, frame = capture_obj.read()
        
        frame = cv2.flip(frame, 1, 0)

       
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

       
        for (x, y, w, h) in faces:
           
            roi_color = frame[y-90:y+h+70, x-50:x+w+50]

            
            cv2.imwrite(save_loc, roi_color)
            curr_time = time.time()

            if curr_time - prev_time >= 3:
                img = cv2.imread(save_loc)
                if img is not None:
                    resize_img(save_loc)

                    min_dist, identity, registered = find_face_realtime(
                        save_loc, database, model, threshold)

                    if min_dist <= threshold and registered:
                        text = 'Welcome ' + identity
                        print('Welcome ' + identity + '!')
                    else:
                        text = 'Unknown user'
                        print('Unknown user' + ' detected !')
                    print('distance:' + str(min_dist))
                prev_time = time.time()
            cv2.rectangle(frame, (x-10, y-70),
                          (x+w+20, y+h+40), (15, 175, 61), 4)
            cv2.putText(frame, text, (50, 50), font, 1.8, (158, 11, 40), 3)


        cv2.imshow('frame', frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    capture_obj.release()
    cv2.destroyAllWindows()



def find_face_realtime(image_path, database, model, threshold):
    encoding = img_to_encoding(image_path, model)
    registered = False
    min_dist = 99999
    identity = 'Unknown Person'
    
    for name in database:
        
        dist = np.linalg.norm(np.subtract(database[name], encoding))
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > threshold:
        registered = False
    else:
        registered = True
    return min_dist, identity, registered
