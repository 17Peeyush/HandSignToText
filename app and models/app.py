import tkinter as tk
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
import math
from keras.models import model_from_json
import PredictionFin as PFIN

mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

class App:

    def __init__(self):
        self.cam= cv2.VideoCapture(0)
        self.initilizeCam()
        self.window=tk.Tk()
        self.window.title("Hand Sign to text")
        self.window.geometry("1920x1080")
        self.current_sentence=""
        self.current_word=""
        self.current_predicted_alphabet=""
        self.predicted_value=[]
        self.initial_space = True
        self.camera = tk.Label(self.window)
        self.camera.place(x=10, y=10, width=1080, height=720)
        self.predicted_panel_text=tk.Label(self.window)
        self.predicted_panel_text.place(x=1150, y=50)
        self.predicted_panel_text.config(text="Predicted Alphabet", font=("Courier", 20))
        self.predicted_panel = tk.Label(self.window)
        self.predicted_panel.place(x=1150, y=100)
        self.current_word_panel_text=tk.Label(self.window)
        self.current_word_panel_text.place(x=1150,y=250)
        self.current_word_panel_text.config(text="Current Word",font=("Courier", 20))
        self.current_word_panel = tk.Label(self.window)
        self.current_word_panel.place(x=1150, y=300)
        self.current_sentence_panel_text = tk.Label(self.window)
        self.current_sentence_panel_text.place(x=1150, y=400)
        self.current_sentence_panel_text.config(text="Current Sentence", font=("Courier", 20))
        self.current_sentence_panel = tk.Label(self.window)
        self.current_sentence_panel.place(x=1150, y=450)
        self.bt = tk.Button(self.window, text="Reset", command=self.resetAll, height=0, width=20)
        self.bt.place(x=1150, y=600)
        self.window.bind("<Key>", self.correctAns)
        self.getVideo()

    def resetAll(self):
        self.current_predicted_alphabet=""
        self.current_word=""
        self.current_sentence=""

    def correctAns(self,event):
        # print(event.keysym,"pressed")
        if str(event.keysym) == "space":
            if self.initial_space:
                self.initial_space= False
            else:
                self.current_sentence+=" "
            self.current_sentence += self.current_word
            self.current_predicted_alphabet = ""
            self.current_word = ""
            self.predicted_value = []

        elif str(event.keysym) == "Up":
            self.current_word+=self.current_predicted_alphabet
        # elif str(event.keysym) == "Escape":
        #     exit()

    def initilizeCam(self):
        # Setting capture window size
        # 3->width
        self.cam.set(3, 1080)
        # 4->height
        self.cam.set(4, 720)

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

    def getans(self):
        values = [0] * 26
        for ch in self.predicted_value:
            values[ord(ch) - 65] += 1
        idx = -1
        max = -1
        for i in range(0, 26):
            if (values[i] > max):
                max = values[i]
                idx = i
        return chr(65 + idx)

    def getVideo(self):
        while True:
            success, frame = self.cam.read()
            if success:
                frame = cv2.flip(frame, 1)
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                # cv2.rectangle(image, start_point, end_point, color, thickness)
                display = cv2.rectangle(frame.copy(), (800, 100), (1150, 450), (0, 255, 0), 2)
                current_frame = ImageTk.PhotoImage(image=Image.fromarray(display))
                self.camera.imgtk = current_frame
                self.camera.config(image=current_frame)
                ROI = frame[100:450, 800:1150].copy()
                # imgRGB = cv2.cvtColor(ROI, cv2.COLOR_BGR2RGB)
                result = hands.process(ROI)
                blank_image = np.zeros(shape=[350, 350, 3], dtype=np.uint8)
                # cv2.imshow("ROI",ROI)
                keypressed = cv2.waitKey(5)
                if result.multi_hand_landmarks!=None:
                    # print("detected")
                    # feature_map = {}
                    for handLms in result.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(blank_image, handLms, mp_hands.HAND_CONNECTIONS)
                        feature_map = {}
                        for id, lm in enumerate(handLms.landmark):
                            # print(id, lm)
                            h, w, c = ROI.shape  # height, width, channels
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
                        ans = PFIN.prediction(feature_map, blank_image)
                        self.predicted_value.append(ans)
                        if (len(self.predicted_value) >= 10):
                            self.current_predicted_alphabet = self.getans()
                            self.predicted_value = []
                else:
                    # print("undetected")
                    self.current_predicted_alphabet=""
                # cv2.imshow("blank",blank_image)
                # elif keypressed & keypressed == 32:  # cv2.waitkey(delay (milliseconds)) space
                #     self.current_sentence+=" "
                #     print("space")
                # elif keypressed & 0xFF == 13:  # cv2.waitkey(delay (milliseconds)) shift
                #     self.current_word+=self.current_predicted_alphabet
                #     print(self.current_word)
                self.predicted_panel.config(text=self.current_predicted_alphabet, font=("Courier", 40))
                self.current_word_panel.config(text=self.current_word, font=("Courier", 30))
                self.current_sentence_panel.config(text=self.current_sentence, font=("Courier", 30))
            self.window.update()

print("obj created")
obj=App()
obj.window.mainloop()
