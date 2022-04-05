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
        self.camera = tk.Label(self.window)
        self.camera.place(x=10, y=10, width=1080, height=720)
        self.predicted_panel = tk.Label(self.window)
        self.predicted_panel.place(x=1100, y=100)
        self.getVideo()


    def initilizeCam(self):
        # Setting capture window size
        # 3->width
        self.cam.set(3, 1080)
        # 4->height
        self.cam.set(4, 720)

    def createSquare(img, x, y):
        factor = math.sqrt(2) * 4
        uppercorner = (int(x - factor), int(y - factor))
        lowercorner = (int(x + factor), int(y + factor))
        # cv2.rectangle(image, start_point, end_point, color, thickness)
        cv2.rectangle(img, uppercorner, lowercorner, (255, 0, 0), -1)

    def createTriangle(img, x, y):
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
                cv2.imshow("ROI",ROI)
                if result.multi_hand_landmarks!=None:
                    print("detected")
                else:
                    print("undetected")
            self.window.update()

print("obj created")
obj=App()
obj.window.mainloop()
