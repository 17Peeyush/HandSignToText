import cv2
import mediapipe as mp
import numpy as np
import math
mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def createSquare(img,x,y):
    factor = math.sqrt(2)*4
    uppercorner=(int(x-factor),int(y-factor))
    lowercorner=(int(x+factor),int(y+factor))
    # cv2.rectangle(image, start_point, end_point, color, thickness)
    cv2.rectangle(img,uppercorner,lowercorner,(255,0,0),-1)

def createTriangle(img,x,y):

    vertices = np.array([[x-4, y], [x+4, y-4], [x+4, y+4]], np.int32)
    pts = vertices.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=(0, 175, 255), thickness=3)
    # fill it
    cv2.fillPoly(img, [pts], color=(0, 175, 255))

cam = cv2.VideoCapture(0)
#Setting capture window size
# 3->width
cam.set(3,1080)
#4->height
cam.set(4,720)

hands = mp_hands.Hands()

if not cam.isOpened():
  print ("Could not open cam")
  exit()

tempcount = 0
while True:
    success, frame = cam.read()
    if success:
        frame = cv2.flip(frame, 1)
        # cv2.rectangle(image, start_point, end_point, color, thickness)
        display = cv2.rectangle(frame.copy(), (800, 100), (1150, 450), (0, 255, 0), 2)
        # [(y1:y2),(x1:x2)]
        ROI = frame[100:450, 800:1150].copy()
        imgRGB= cv2.cvtColor(ROI, cv2.COLOR_BGR2RGB)
        result= hands.process(imgRGB)
        blank_image = np.zeros(shape=[350, 350, 3], dtype=np.uint8)
        if(result.multi_hand_landmarks!=None):
            for handLms in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(blank_image, handLms, mp_hands.HAND_CONNECTIONS)
                for id, lm in enumerate(handLms.landmark):
                    # print(id, lm)
                    h, w, c=imgRGB.shape #height, width, channels
                    cx, cy=int(lm.x*w), int(lm.y*h) # center coordinates
                    if (id == 5 or id == 9 or id== 13 or id== 17):
                        cv2.circle(blank_image, (cx, cy), 5, (131, 245, 44), -1) #florant green for MCP
                    elif (id == 8 or id == 12 or id == 16 or id == 20):  # fingerpits triangle yellow
                        createTriangle(blank_image, cx, cy)
                    elif (id==1 or id== 2 or id==3): #giving preference to thumb
                        cv2.circle(blank_image, (cx, cy), 5, (30, 144, 255), -1)
                    elif(id==4): #thumb tip
                        createSquare(blank_image,cx,cy)
                        # cv2.circle(image, center_coordinates, radius, color, thickness)
                # mp_drawing.draw_landmarks(blank_image,handLms, mp_hands.HAND_CONNECTIONS)
                keypressed = cv2.waitKey(5)
                if keypressed & 0xFF == 27:  # cv2.waitkey(delay (milliseconds))
                    exit()
                if keypressed & 0xFF == ord('a'):
                    cv2.imwrite(str(tempcount) + '.jpg', blank_image)
                    tempcount=tempcount+1
                    if tempcount==251:
                        tempcount=0
                    print("a pressed", tempcount)

        # grayImage = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        # blur = cv2.GaussianBlur(grayImage, (5, 5), 2)
        # th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        # ret, test_image = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cv2.imshow('Current Roi', blank_image)
        cv2.imshow("Capture Video", display)
        keypressed = cv2.waitKey(5)
        if keypressed& 0xFF == 27: # cv2.waitkey(delay (milliseconds))
            break

cam.release()
cv2.destroyAllWindows()