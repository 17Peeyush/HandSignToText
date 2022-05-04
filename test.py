import  cv2
import  os
import mediapipe as mp
import numpy as np
import math
import PredictionFin as PFIN
class test:


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


    def checkCam(self):
        if not self.cam.isOpened():
            print("Could not open cam")
            exit()
        # Setting capture window size
        # 3->width
        self.cam.set(3, 1080)
        # 4->height
        self.cam.set(4, 720)
        print("Camera successfully initialized...")

    def initialize_result(self):
        for _ in range(0,26):
            temp=[0]*26
            self.result.append(temp)


    def __init__(self):
        print("Object created")
        self.alphacount = [0]*26
        self.cam = cv2.VideoCapture(0)
        self.checkCam()
        self.result=[]
        self.initialize_result()

    def tester(self):
        # , mp_drawing, mp_hands, hands
        mp_drawing = mp.solutions.drawing_utils
        # mp_drawing_styles = mp.solutions.drawing_styles
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands()
        while True:
            success, frame = self.cam.read()
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
                        feature_map = {}
                        for id, lm in enumerate(handLms.landmark):
                            # print(id, lm)
                            h, w, c = imgRGB.shape  # height, width, channels
                            cx, cy = int(lm.x * w), int(lm.y * h)  # center coordinates
                            feature_map[id] = [cx, cy]
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
                                print(chr(i), "pressed",self.alphacount[i-97])
                                ans = PFIN.prediction(feature_map, blank_image)
                                print(ans)
                                self.alphacount[i-97]+=1
                                self.result[i-97][ord(ans)-65]+=1
                if keypressed & 0xFF == 27:  # cv2.waitkey(delay (milliseconds))
                    break
                cv2.imshow('Filter Roi', blank_image)
                cv2.imshow("Capture Video", display)
    def releaseResource(self):
        print(self.alphacount)
        for arr in self.result:
            print(arr)
        self.cam.release()
        cv2.destroyAllWindows()


c = test()
c.tester()
c.releaseResource()
