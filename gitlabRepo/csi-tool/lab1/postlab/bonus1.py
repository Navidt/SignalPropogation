from sense_hat import SenseHat
from time import sleep
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import CircularOutput
from libcamera import controls
import cv2
picam2=Picamera2()  ## Create a camera object

# text
font = cv2.FONT_HERSHEY_SIMPLEX 
fontScale = 1
color = (255, 0, 0) 
thickness = 2

# initialize SenseHat and set temperature and humidity
sense = SenseHat()
sense.clear()
t = sense.get_temperature()
h = sense.get_humidity()


dispW=1280
dispH=720
picam2.preview_configuration.main.size= (dispW,dispH)  ## 1280 cols, 720 rows. Can also try smaller size of frame as (640,360) and the largest (1920,1080)
picam2.preview_configuration.main.format= "RGB888"
picam2.preview_configuration.align() ## aligns the size to the closest standard format
picam2.preview_configuration.controls.FrameRate=30 ## set the number of frames per second, this is set as a request, the actual time it takes for processing each frame and rendering a frame can be different

picam2.configure("preview")

faceCascade=cv2.CascadeClassifier("./eye.xml")
flag = False

while True:
    picam2.start()
    frame=picam2.capture_array() ## frame is a large 2D array of rows and cols and at intersection of each point there is an array of three numbers for RGB i.e. [R,G,B] where RGB value ranges from 0 to 255

    frameGray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(frameGray,1.3,5)

    face_count = 0

    for face in faces:
        x,y,w,h=face
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0),3)
        face_count += 1

    image = cv2.putText(frame, 'detecting eyes...', (30,30), font,  fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow("Camera Frame", frame)  
 
    key=cv2.waitKey(1) & 0xFF

    if key ==ord(" "):
        cv2.imwrite("frame-" + str(time.strftime("%H:%M:%S", time.localtime())) + ".jpg", frame)
    if key == ord("q"): ## stops for 1 ms to check if key Q is pressed
        break

cv2.destroyAllWindows()

    