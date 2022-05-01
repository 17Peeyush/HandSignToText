import  cv2
import  os
import mediapipe as mp
import numpy as np
import math

class collect:

    cam = None
    directory = None
    alphacount = []

    def createSquare(self,img, x, y):
        factor = math.sqrt(2) * 4
        uppercorner = (int(x - factor), int(y - factor))
        lowercorner = (int(x + factor), int(y + factor))
        # cv2.rectangle(image, start_point, end_point, color, thickness)
        cv2.rectangle(img, uppercorner, lowercorner, (255, 0, 0), -1)

    def createTriangle(self,img, x, y):
        vertices = np.array([[x - 4, y], [x + 4, y - 4], [x + 4, y + 4]], np.int32)
        pts = vertices.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(0, 175, 255), thickness=3)
        # fill it
        cv2.fillPoly(img, [pts], color=(0, 175, 255))


    def checkCam(self, cam):
        if not cam.isOpened():
            print("Could not open cam")
            exit()
        # Setting capture window size
        # 3->width
        cam.set(3, 1080)
        # 4->height
        cam.set(4, 720)
        print("Camera successfully initialized...")

    def initializeOS(self):
        os.chdir(directory)
        if not os.path.exists("data"):
            os.makedirs("data")
        if not os.path.exists("data/train"):
            os.makedirs("data/train")
        if not os.path.exists("data/test"):
            os.makedirs("data/test")
        for i in range(65, 91):
            if not os.path.exists("data/train"+chr(i)):
                os.makedirs("data/train/" + chr(i))
            if not os.path.exists("data/test"+chr(i)):
                os.makedirs("data/test/" + chr(i))
        print("Directories Created Successfully...")

    def __init__(self):
        print("Object created")
        global cam, directory, alphacount
        alphacount = [0]*26
        print(alphacount)
        cam = cv2.VideoCapture(0)
        directory = r'D:\playground\Major Project (Hand sign)\SubGroup'
        self.checkCam(cam)
        self.initializeOS()

    def generateDataSet(self):
        global cam, directory, alphacount
        # , mp_drawing, mp_hands, hands
        mp_drawing = mp.solutions.drawing_utils
        # mp_drawing_styles = mp.solutions.drawing_styles
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands()
        while True:
            success, frame = cam.read()
            if success:
                frame = cv2.flip(frame, 1)
                # cv2.rectangle(image, start_point, end_point, color, thickness)
                display = cv2.rectangle(frame.copy(), (800, 100), (1150, 450), (0, 255, 0), 2)
                # [(y1:y2),(x1:x2)]
                ROI = frame[100:450, 800:1150].copy()
                imgRGB = cv2.cvtColor(ROI, cv2.COLOR_BGR2RGB)
                result = hands.process(imgRGB)
                blank_image = np.zeros(shape=[350, 350, 3], dtype=np.uint8)
                keypressed = cv2.waitKey(5)
                if(result.multi_hand_landmarks!=None):
                    for handLms in result.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(blank_image, handLms, mp_hands.HAND_CONNECTIONS)
                        for id, lm in enumerate(handLms.landmark):
                            # print(id, lm)
                            h, w, c = imgRGB.shape  # height, width, channels
                            cx, cy = int(lm.x * w), int(lm.y * h)  # center coordinates
                            if (id == 5 or id == 9 or id == 13 or id == 17):
                                cv2.circle(blank_image, (cx, cy), 5, (131, 245, 44), -1)  # florant green for MCP
                            elif (id == 8 or id == 12 or id == 16 or id == 20):  # fingerpits triangle yellow
                                self.createTriangle(blank_image, cx, cy)
                            elif (id == 1 or id == 2 or id == 3):  # giving preference to thumb
                                cv2.circle(blank_image, (cx, cy), 5, (30, 144, 255), -1)
                            elif (id == 4):  # thumb tip
                                self.createSquare(blank_image, cx, cy)
                                # cv2.circle(image, center_coordinates, radius, color, thickness)
                        for i in range(97, 123):
                            if keypressed & 0xFF == ord(chr(i)):
                                print(chr(i), "pressed")
                                tempcount = alphacount[i-97]
                                alphacount[i-97]+=1
                                print(i,"  ",tempcount)
                                if(tempcount < 1000):
                                    tempdirectory='data/train/'+chr(i)
                                    os.chdir(tempdirectory)
                                    cv2.imwrite(str(tempcount)+'.jpg',blank_image)
                                else:
                                    tempdirectory = 'data/test/' + chr(i)
                                    os.chdir(tempdirectory)
                                    cv2.imwrite(str(tempcount-1000)+'.jpg', blank_image)
                                os.chdir(directory)
                if keypressed & 0xFF == 27:  # cv2.waitkey(delay (milliseconds))
                    break
                cv2.imshow('Filter Roi', blank_image)
                cv2.imshow("Capture Video", display)
    def releaseResource(self):
        global cam
        cam.release()
        cv2.destroyAllWindows()


c = collect()
c.generateDataSet()
c.releaseResource()
