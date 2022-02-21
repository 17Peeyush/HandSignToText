import  cv2
import  os

class collect:

    cam = None
    directory = None
    alphacount = []

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
        directory = r'D:\playground\Major Project (Hand sign)'
        self.checkCam(cam)
        self.initializeOS()

    def generateDataSet(self):
        global cam, directory, alphacount
        while True:
            success, frame = cam.read()
            if success:
                frame = cv2.flip(frame, 1)
                # cv2.rectangle(image, start_point, end_point, color, thickness)
                display = cv2.rectangle(frame.copy(), (800, 100), (1150, 450), (0, 255, 0), 2)
                # [(y1:y2),(x1:x2)]
                ROI = frame[100:450, 800:1150].copy()
                grayImage = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(grayImage, (5, 5), 2)
                th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                ret, filter_image = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                cv2.imshow('Filter Roi', filter_image)
                cv2.imshow("Capture Video", display)
                keypressed = cv2.waitKey(5)
                for i in range(97, 123):
                    if keypressed & 0xFF == ord(chr(i)):
                        print(chr(i), "pressed")
                        tempcount = alphacount[i-97]
                        alphacount[i-97]+=1
                        print(i,"  ",tempcount)
                        if(tempcount < 500):
                            tempdirectory='data/train/'+chr(i)
                            os.chdir(tempdirectory)
                            cv2.imwrite(str(tempcount)+'.jpg',filter_image)
                        else:
                            tempdirectory = 'data/test/' + chr(i)
                            os.chdir(tempdirectory)
                            cv2.imwrite(str(tempcount-500)+'.jpg', filter_image)
                        os.chdir(directory)
                if keypressed & 0xFF == 27:  # cv2.waitkey(delay (milliseconds))
                    break

    def releaseResource(self):
        global cam
        cam.release()
        cv2.destroyAllWindows()


c = collect()
c.generateDataSet()
c.releaseResource()
