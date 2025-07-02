from math import ceil

import cv2
import numpy as np

def splitBox(img):
    rows = np.vsplit(img,10)
    print(rows)

def rectCountour(coutours):
    rectCon = []
    for i in coutours:
        area = cv2.contourArea(i)
        #print(area)
        if area > 100:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i,0.03*peri, True) # true la tim duong khep kin
            #neu la hinh chu nhat
            if len(approx) == 4:
                rectCon.append(i)
    rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True) # sap xep theo dien tich giam dan
    return rectCon

def rectCountourid(coutours):
    rectCon = []
    for i in coutours:
        area = cv2.contourArea(i)
        if area > 500:
            print(area)
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i,0.03*peri, True) # true la tim duong khep kin
            #neu la hinh chu nhat
            if len(approx) == 4:
                rectCon.append(i)
    rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True) # sap xep theo dien tich giam dan
    return rectCon


def getConnerPoints(cont):
    #tu contour chuyen ve toa do 4 dinh
    peri = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.03 * peri, True)  # true la tim duong khep kin
    return approx

def reorder(myPoints):
    myPoints = myPoints.reshape(4,2)
    myNewPoints = np.zeros((4,1,2), np.int32)
    add = myPoints.sum(1)
    myNewPoints[0] = myPoints[np.argmin(add)]
    myNewPoints[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myNewPoints[1] = myPoints[np.argmin(diff)]
    myNewPoints[2] = myPoints[np.argmax(diff)]
    #print(f'mynewpoint',myNewPoints)
    return myNewPoints


def sort_contours(cnts, method="left-to-right"):
    return cnts

def ResizeImage(img, height=800):
    rat = height / img.shape[0]
    width = int(rat * img.shape[1])
    dst = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return dst



def splitBoxes(img, top_left_x=0,top_left_y = 0 ):
    #print("x,y", top_left_x, top_left_y)
    box_img = img[15:img.shape[0]-5,25:img.shape[1]-10]
    box_img = ResizeImage(box_img,height=200)
    #print(box_img.shape[0])
    #print(box_img.shape[1])
    offset2 = ceil(box_img.shape[0] / 10)

    list_answers =[]
    list_circle = []
    offset = 34
    for j in range(10):
        list_answers.append(box_img[j * offset2:(j + 1) * offset2, :])
        for i in range(4):
            bubble_choice = list_answers[j][:, i * offset: (i + 1) * offset]
            bubble_choice = cv2.threshold(bubble_choice, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            bubble_choice = cv2.resize(bubble_choice, (28, 28), cv2.INTER_AREA)
            bubble_choice = bubble_choice.reshape((28, 28, 1))
            #cv2.imshow(f'{j}{i}', bubble_choice)
            list_circle.append(bubble_choice)
    #print(len(list_circle))
    return  list_circle









    '''

    :param img:
    :return:
     rows = (np.vsplit(img,10))
    boxes = []
    for r in rows:
        cols = np.hsplit(r,4)
        for box in cols:
            boxes.append(box)
    return boxes
    '''





def splitBoxesIdTest(img):
    rows = (np.vsplit(img,10))
    boxes = []
    for r in rows:
        cols = np.hsplit(r,3)
        for box in cols:
            boxes.append(box)
    return boxes

def splitBoxesIdStudent(img):
    rows = (np.vsplit(img,10))
    boxes = []
    for r in rows:
        cols = np.hsplit(r,6)
        for box in cols:
            boxes.append(box)
    return boxes



def showAnswer(img, myIndex , answer, mypixcelCoords ):
    secW = int (img.shape[1]/4)
    secH = int (img.shape[0]/10)
    for x in range(0,10):
        numb = myIndex[x]
        an = answer[x]
        for i in range(0,4):
            c_x = mypixcelCoords[x][i][0]+20
            c_y = mypixcelCoords[x][i][1]+20
            if numb[i] == 1 and an == i and sum(numb) ==1:
                #dung
                cX = (i * secW) + secW // 2
                cY = (x * secH) + secH // 2
                cv2.circle(img, (c_x,c_y ), 10, (0, 255, 0), 3)
            elif numb[i] == 1:
                #sai
                cX = (i * secW) + secW // 2
                cY = (x * secH) + secH // 2
                #cv2.circle(img, (cX, cY), 10, (0, 0, 255), 3)
                cv2.circle(img, (c_x, c_y), 10, (0, 0, 255), 3)
            elif numb[i] == 1 and an == i:
                cX = (i * secW) + secW // 2
                cY = (x * secH) + secH // 2
                #cv2.circle(img, (cX, cY), 10, (0, 0, 255),3)
                cv2.circle(img, (c_x, c_y), 10, (0, 0, 255), 3)
    return img



