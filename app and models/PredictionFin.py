import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import math
import operator

#IJY model loading
json_file = open("IJY.json", "r")
model_json = json_file.read()
json_file.close()
model_IJY = model_from_json(model_json)
# load weights into new model
model_IJY.load_weights("IJY.h5")
print("Loaded IJY model from disk")

#FW model loading
json_fileFW = open("F_W.json", "r")
model_json = json_fileFW.read()
json_fileFW.close()
model_FW = model_from_json(model_json)
model_FW.load_weights("F_W.h5")
print("Loaded FW model from disk")

#BC model loading
json_fileFW = open("B_C.json", "r")
model_json = json_fileFW.read()
json_fileFW.close()
model_BC = model_from_json(model_json)
model_BC.load_weights("B_C.h5")
print("Loaded BC model from disk")

#HKRUV model loading
json_file = open("HK_R_UV.json", "r")
model_json = json_file.read()
json_file.close()
model_HKRUV = model_from_json(model_json)
# load weights into new model
model_HKRUV.load_weights("HK_R_UV.h5")
print("Loaded HKRUV model from disk")

#DGLPXZ model loading
json_file = open("DGL_PXZ.json", "r")
model_json = json_file.read()
json_file.close()
model_DGLPXZ = model_from_json(model_json)
# load weights into new model
model_DGLPXZ.load_weights("DGL_PXZ.h5")
print("Loaded DG model from disk")

#AEMNOST model loading
json_file = open("MNS.json", "r")
model_json = json_file.read()
json_file.close()
model_MNS = model_from_json(model_json)
# load weights into new model
model_MNS.load_weights("MNS.h5")
print("Loaded MNS model from disk")

def predictDGLPXZ(blank_image):
    img_pred = image.img_to_array(blank_image)
    img_pred = np.expand_dims(img_pred, axis=0)
    rslt = model_DGLPXZ.predict(img_pred)
    prediction = {
        'D': rslt[0][0],
        'G': rslt[0][1],
        'L': rslt[0][2],
        'P': rslt[0][3],
        'X': rslt[0][4],
        'Z': rslt[0][5],
    }
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    return(prediction[0][0])

def predictIJY(blank_image):
    img_pred = image.img_to_array(blank_image)
    img_pred = np.expand_dims(img_pred, axis=0)
    rslt = model_IJY.predict(img_pred)
    if(rslt[0][0]==1):
        return 'I'
    elif(rslt[0][1]==1):
        return 'J'
    else:
        return 'Y'
def predictHKRUV(feature_map,blank_image):
    img_pred = image.img_to_array(blank_image)
    img_pred = np.expand_dims(img_pred, axis=0)
    rslt = model_HKRUV.predict(img_pred)
    prediction = {
        'H': rslt[0][0],
        'K': rslt[0][1],
        'R': rslt[0][2],
        'U': rslt[0][3],
        'V': rslt[0][4]
    }
    # print(rslt)
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    predval= prediction[0][0]
    # if predval=='U':
    #     if feature_map[11][0]<feature_map[7][0]:
    #         return 'R'
    #     return 'U'
    return predval

def predictFandW(blank_image):
    img_pred = image.img_to_array(blank_image)
    img_pred = np.expand_dims(img_pred, axis=0)
    rslt = model_FW.predict(img_pred)
    # print("first", rslt[0][0])
    # print("sec", rslt[0][1])
    if (rslt[0][0] == 1):
        return ('F')
    else:
        return ('W')

def predictBandC(blank_image,feature_map):
    img_pred = image.img_to_array(blank_image)
    img_pred = np.expand_dims(img_pred, axis=0)
    rslt = model_BC.predict(img_pred)
    # print("first", rslt[0][0])
    # print("sec", rslt[0][1])
    if (rslt[0][0] == 1):
        if( feature_map[4][0]<feature_map[5][0]):
            return ('C')
        return ('B')
    else:
        return ('C')

def predictAEM_N_OST(blank_image,feature_map):
    if feature_map[4][1]<feature_map[5][1] and feature_map[4][0]>feature_map[6][0] and feature_map[4][0]<feature_map[10][0]:
        return 'T'
    elif feature_map[4][0]>feature_map[2][0] and feature_map[4][1]>feature_map[0][1]:
        return 'Q'
    elif feature_map[4][1] > feature_map[9][1] and feature_map[4][0]>feature_map[5][0]:
        return 'E'
    elif feature_map[4][0]<feature_map[6][0]:
        if feature_map[4][1]<feature_map[6][1] and feature_map[4][1]<feature_map[10][1]:
            return 'A'
        return 'O'
    # return 'otherzero'
    else:
        img_pred = image.img_to_array(blank_image)
        img_pred = np.expand_dims(img_pred, axis=0)
        rslt = model_MNS.predict(img_pred)
        prediction = {
            'M': rslt[0][0],
            'N': rslt[0][1],
            'S': rslt[0][2]
        }
        # print(rslt)
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        return prediction[0][0]
        return("MNS")
        pass


def fingercount(feature_map):
    indexFinger = 0
    middleFinger = 0
    ringFinger = 0
    littleFinger = 0
    fingerIdentification=[False,False,False,False] #[index,middle,ring,little]
    if feature_map[5][1] > feature_map[8][1] and feature_map[6][1] > feature_map[8][1]:
        indexFinger = 1
        fingerIdentification[0]=True
    if feature_map[9][1] > feature_map[12][1] and feature_map[10][1] > feature_map[12][1]:
        middleFinger = 1
        fingerIdentification[1]=True
    if feature_map[13][1] > feature_map[16][1] and feature_map[14][1] > feature_map[16][1]:
        ringFinger = 1
        fingerIdentification[2]=True
    if feature_map[17][1] > feature_map[20][1] and feature_map[18][1] > feature_map[20][1]:
        littleFinger = 1
        fingerIdentification[3]=True
    fingers=indexFinger + middleFinger + ringFinger + littleFinger
    # print(fingers, "fingers")
    return [fingers,fingerIdentification]

def prediction(feature_map,blank_image)->str:
    count,fingerIdentification=fingercount(feature_map)
    # print('count:',count)
    if (count==4):
        # print('c4->B')
        return predictBandC(blank_image,feature_map)
        # return 'B'
    elif(count==3):
        # print('c3->fw')
        return predictFandW(blank_image)
    elif(count==2):
        return predictHKRUV(feature_map,blank_image)
    elif(count==1):
        if(fingerIdentification[3]):
            return predictIJY(blank_image)
        else:
            return predictDGLPXZ(blank_image)

    else:
        return predictAEM_N_OST(blank_image,feature_map)
        return str(count)+"fingers@"