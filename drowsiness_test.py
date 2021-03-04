import cv2
import dlib
ld = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fd = dlib.get_frontal_face_detector()

cam = cv2.VideoCapture(0)


while 1:
    _,frame = cam.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = fd(gray,1)
    for face in faces:
        print("s")
        face_landmarks = fd(gray,face)
        for n in range(1,68):
            x,y = face_landmarks.part(n).x,face_landmarks.part(n).y
            cv2.circle(frame,(x,y),1,(255,255,0),-1)
    cv2.imshow("cam",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  

cam.release() 
cv2.destroyAllWindows() 