import math
import uuid
from collections import defaultdict

import cv2
import imutils as imutils
import numpy as np

import utils
from model import CNN_Model
from keras.models import load_model


def get_x(s):
    return s[1][0]


def get_y(s):
    return s[1][1]


def get_h(s):
    return s[1][3]


def get_x_ver1(s):
    s = cv2.boundingRect(s)
    return s[0] * s[1]

def crop_image(img):
    # convert image from BGR to GRAY to apply canny edge detection algorithm
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # remove noise by blur image
    blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
    # apply canny edge detection algorithm
    img_canny = cv2.Canny(blurred, 90, 200)
    # find contours
    cnts = cv2.findContours(img_canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    ans_blocks = []
    x_old, y_old, w_old, h_old = 0, 0, 0, 0
    # ensure that at least one contour was found
    if len(cnts) > 0:
        # sort the contours according to their size in descending order
        cnts = sorted(cnts, key=get_x_ver1)

        # loop over the sorted contours
        for i, c in enumerate(cnts):
            x_curr, y_curr, w_curr, h_curr = cv2.boundingRect(c)

            if w_curr * h_curr > 400:
                # check overlap contours
                check_xy_min = x_curr * y_curr - x_old * y_old
                check_xy_max = (x_curr + w_curr) * (y_curr + h_curr) - (x_old + w_old) * (y_old + h_old)

                # if list answer box is empty
                if len(ans_blocks) == 0:
                    ans_blocks.append(
                        (gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr], [x_curr, y_curr, w_curr, h_curr]))
                    cv2.drawContours(img, [c], 0, (0, 255, 0), 2)
                    # update coordinates (x, y) and (height, width) of added contours
                    x_old = x_curr
                    y_old = y_curr
                    w_old = w_curr
                    h_old = h_curr
                elif check_xy_min > 200 and check_xy_max > 200:
                    ans_blocks.append(
                        (gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr], [x_curr, y_curr, w_curr, h_curr]))
                    # update coordinates (x, y) and (height, width) of added contours
                    cv2.drawContours(img, [c], 0, (0, 255, 255), 2)
                    x_old = x_curr
                    y_old = y_curr
                    w_old = w_curr
                    h_old = h_curr

        # sort ans_blocks according to x coordinate
        sorted_ans_blocks = sorted(ans_blocks, key=get_x)
        print(len(ans_blocks))
        return sorted_ans_blocks


def timhinh2(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Áp dụng ngưỡng để chuyển ảnh xám sang ảnh nhị phân
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    #Tìm các contour trong ảnh nhị phân
    #cv2.imshow('th',binary )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #Biến để lưu hình vuông đen lớn nhất
    largest_square = None
    largest_area = 0
    # Lọc và tìm hình vuông đen lớn nhất
    for contour in contours:
        # Tìm hình chữ nhật bao quanh contour
        x, y, w, h = cv2.boundingRect(contour)
        # Kiểm tra xem nó có nằm dưới số 9 không (giả sử số 9 nằm ở tọa độ y=100)
        if y > 200 and y<280:
            # Tính diện tích của contour
            area = cv2.contourArea(contour)
            # Kiểm tra xem đây có phải là hình vuông và có diện tích lớn nhất không
            if area > largest_area and abs(w - h) < 10:
                largest_square = contour
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
                largest_area = area
    # Vẽ hình vuông lớn nhất trên ảnh gốc
    if largest_square is not None:
        x, y, w, h = cv2.boundingRect(largest_square)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # Tính tọa độ tâm
        center_x = x + w // 2
        center_y = y + h // 2
        # Vẽ một điểm ở tâm hình vuông
        cv2.circle(image, (center_x, center_y), 1, (0, 0, 255), -1)
        print(f"Tọa độ tâm của hình vuông: ({center_x}, {center_y})")
    return (center_x, center_y)
    #cv2.imshow('Largest Square', image)


def timhinh3(image, loca):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Áp dụng ngưỡng để chuyển ảnh xám sang ảnh nhị phân
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    #Tìm các contour trong ảnh nhị phân
    #cv2.imshow('th',binary )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #Biến để lưu hình vuông đen lớn nhất
    largest_square = None
    largest_area = 0
    # Lọc và tìm hình vuông đen lớn nhất
    for contour in contours:
        # Tìm hình chữ nhật bao quanh contour
        x, y, w, h = cv2.boundingRect(contour)
        # Kiểm tra xem nó có nằm dưới số 9 không (giả sử số 9 nằm ở tọa độ y=100)
        if y > loca[1]-10 and y<loca[1]+10 and x>loca[0]-10 and x<loca[0]+10 :
            # Tính diện tích của contour
            area = cv2.contourArea(contour)
            # Kiểm tra xem đây có phải là hình vuông và có diện tích lớn nhất không
            if area > largest_area and abs(w - h) < 10:
                largest_square = contour
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
                largest_area = area
    # Vẽ hình vuông lớn nhất trên ảnh gốc
    if largest_square is not None:
        x, y, w, h = cv2.boundingRect(largest_square)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
        # Tính tọa độ tâm
        center_x = x + w // 2
        center_y = y + h // 2
        # Vẽ một điểm ở tâm hình vuông
        #cv2.circle(image, (center_x, center_y), 1, (0, 0, 255), -1)
        #print(f"Tọa độ tâm của hình vuông nhỏ: ({center_x}, {center_y})")
    return (center_x, center_y)
    #cv2.imshow('Largest Square', image)






def ResizeImage(img, height=800):
    rat = height / img.shape[0]
    width = int(rat * img.shape[1])
    dst = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return dst


def find_anchor2(img):
    anchors = []
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold the image to create a binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #cv2.imshow('the', thresh)
    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(img, contours, -1, (225,0,0), 3)
    #cv2.imshow("ancho", img)
    # Iterate through each contour and find the rectangular contour
    for cnt in contours:
        # Find the perimeter of the contour
        perimeter = cv2.arcLength(cnt, True)
        # Approximate the contour to a polygon
        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
        # Check if the polygon is a square or rectangle
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            cv2.drawContours(img, [approx], 0, (0, 255, 0), 1)

            aspect_ratio = float(w) / h
            # Check if the aspect ratio is close to 1
            if aspect_ratio >= 0.9 and aspect_ratio <= 1.1:
                # Check if the square is dark
                roi = gray[y:y + h, x:x + w]
                mean = cv2.mean(roi)[0]
                if mean < 100:
                    print(mean)
                    anchors.append(approx.reshape(-1,2))
                    # Draw the square on the image
                    #cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)

    # Display the image
    cv2.imshow('image', img)
    cv2.waitKey()
    return anchors

def FindAnchors(img, Area=np.inf, deltaArea=np.inf):
    anchors = []
    anchors2 = []  #lay tat ca, khong can loc, khong can dieu kien
    src_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    src_HSV = cv2.blur(src_HSV, (5, 5))
    edges = cv2.inRange(src_HSV, (0, 0, 0), (255, 255, 50))
    cv2.imshow("edt", edges)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        anchors2.append(approx.reshape(-1,2))
        #cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)
        area = cv2.contourArea(contour)
        #print(area, "dien tich")
        if area > 10:
            if Area == np.inf or deltaArea == np.inf:
                ok = len(approx) == 4 and cv2.isContourConvex(approx)
            else:
                ok = len(approx) == 4 and (area > Area - deltaArea) and (
                            area < Area + deltaArea) and cv2.isContourConvex(
                    approx)
            if ok:
                anchors.append(approx.reshape(-1, 2))


    return anchors2

def ListPointIntersection(contours):
    dst = []
    for i in contours:
        point = PointIntersecsion(i)
        dst.append(point)
    return dst

def PointIntersecsion(points):
    M = cv2.moments(points)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    # giao nhau
    return (cx,cy)

def ClusterPoints(points, distance = 3):
    #gom nhom cac diem lai va tinh khoang cach trung binh sau do tra ve lai cac diem
    dst =[]
    while points:
        count = 1
        p = points[0]
        sumX = p[0]
        sumY = p[1]
        points.pop(0)
        i = 1
        while i<len(points):
            if p == points[i]:
                points.pop(i)
            elif Distance(p, points[i]) < distance:
                sumY += points[i][0]
                sumY += points[i][1]
                count += 1
                points.pop(i)
            else:
                i +=1
        dst.append((int(sumX / count), int(sumY / count)))

    return dst

def Distance(p1,p2) -> float:
    dis  = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
    return math.sqrt(dis)


def find_Point_InRegion(points, condition1,condition2,axis):
    point = np.array([0, 0], dtype=np.float32)
    #axit 0 va 1, 0 theo y , 1 theo truc x
    for i in range(0,len(points)):
        if axis==1:
            #tim point trong khoang denta y
            average = (condition1[1] + condition2[1])/ 2.0
            delta = abs(condition1[1] - condition2[1]) + 5
            print(f'gia tri trung binh',average)
            print(f'dental',delta)
            if points[i][1] == average:
                point = points[i]
                break
            else:
                ok = points[i][1] > (average - delta) and points[i][1] < (average+delta)
                if ok:
                    #lay duoc diem dau tien thoa man va thoat vong lap
                    point = points[i]
                    break
        else:
            #tìm point trong khoảng delta x
            average = (condition1[0] + condition2[0]) / 2.0
            delta = abs(condition1[0] - condition2[0]) + 5
            print(f'gia tri trung binh theo truc ngang', average)
            print(f'dental theo ngang', delta)
            if points[i][0] == average:
                point = points[i]
                break
            else:
                ok = points[i][0] > (average - delta) and points[i][0] < (average + delta)
                if ok:
                    point = points[i]
                    print(point)
                    break
    return point


def adjust_coordinates(coords):
    adjusted_coords = coords[:]
    for i in range(len(adjusted_coords)):
        x1, y1 = adjusted_coords[i]
        for j in range(i + 1, len(adjusted_coords)):
            x2, y2 = adjusted_coords[j]

            if abs(x1 - x2) < 2:
                adjusted_coords[j] = (x1, y2)
            if abs(y1 - y2) < 2:
                adjusted_coords[j] = (x2, y1)

    return adjusted_coords

def sub_rect_image(img_src,indentityRegion):
    rect = cv2.boundingRect(np.array(indentityRegion,dtype=np.int32))
    sub_image = img_src[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]].copy()
    return sub_image

def TranformPoints(points, offset,rat):
    for i in range(len(points)):
        points[i] = (points[i][0]*rat +offset[0], points[i][1]*rat + offset[1])
    return points

def finCircle(imgrect):
    result = []
    for box in imgrect:
        gray = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
        list_circle = utils.splitBoxes(gray, 0, 0)
        result.extend(list_circle)

    return result


def finCircle_ds(imgrect):
    #result = []
    gray = cv2.cvtColor(imgrect, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray', gray)
    box_img = gray[30:gray.shape[0] - 5, 26:gray.shape[1] - 12]
    box_img = ResizeImage(box_img, height=136)
    #cv2.imshow('gray', box_img)
    offset2 = math.ceil(box_img.shape[1] / 2)
    #lay 1/2 hinh
    list_boxds=[]
    box1 = box_img[:, 0:offset2-4]
    box2 = box_img[:, offset2+4:]
    #cv2.imshow('hhhh6666', box2)
    list_boxds.append(box1)
    list_boxds.append(box2)

    offset2 = math.ceil(box_img.shape[0] / 4)
    list_circle = []
    offset = 57
    for itembox in list_boxds:
        list_answers = []
        for j in range(4):
            list_answers.append(itembox[j * offset2:(j + 1) * offset2, :])
            for i in range(2):
                bubble_choice = list_answers[j][:, i * offset: (i + 1) * offset]
                bubble_choice = cv2.threshold(bubble_choice, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

                bubble_choice = cv2.resize(bubble_choice, (28, 28), cv2.INTER_AREA)
                bubble_choice = bubble_choice.reshape((28, 28, 1))
                #cv2.imshow(f'{j}{i}', bubble_choice)
                list_circle.append(bubble_choice)
    #print(len(list_circle))
    return list_circle


def finCircle_tl(imgrect):
    #result = []
    gray = cv2.cvtColor(imgrect, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray', gray)
    box_img = gray[49:gray.shape[0] - 22, 10:gray.shape[1] - 2]
    box_img = ResizeImage(box_img, height=240)
    #print(box_img.shape[0], box_img.shape[1])
    #cv2.imshow('gray', box_img)
    offset2 = math.ceil(box_img.shape[0] / 12)
    list_circle = []
    offset = 26
    list_answers = []
    for j in range(12):
        list_answers.append(box_img[j * offset2:(j + 1) * offset2, :])
        #xu li tung hang, hang dau dau - hang 2 dau ,
        if j==0:
            bubble_choice = list_answers[j][:, 0 * offset: (0 + 1) * offset]
            bubble_choice = cv2.threshold(bubble_choice, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            bubble_choice = cv2.resize(bubble_choice, (28, 28), cv2.INTER_AREA)
            bubble_choice = bubble_choice.reshape((28, 28, 1))
            list_circle.append(bubble_choice)
        elif j==1:
            for i in range(1,3):
                bubble_choice = list_answers[j][:, i * offset: (i + 1) * offset]
                bubble_choice = cv2.threshold(bubble_choice, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                bubble_choice = cv2.resize(bubble_choice, (28, 28), cv2.INTER_AREA)
                bubble_choice = bubble_choice.reshape((28, 28, 1))
                list_circle.append(bubble_choice)
        else:
            for i in range(4):
                bubble_choice = list_answers[j][:, i * offset: (i + 1) * offset]
                bubble_choice = cv2.threshold(bubble_choice, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                bubble_choice = cv2.resize(bubble_choice, (28, 28), cv2.INTER_AREA)
                bubble_choice = bubble_choice.reshape((28, 28, 1))
                list_circle.append(bubble_choice)
                #cv2.imwrite(f'result/tuluna20_1_{j}{i}.png', bubble_choice )
    #print(len(list_circle))
    return list_circle


def finCircle_idtest(imgrect):
    #result = []
    gray = cv2.cvtColor(imgrect, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray', gray)
    box_img = gray[:gray.shape[0] - 4, 10:]
    box_img = ResizeImage(box_img, height=300)
    #print(box_img.shape[0], box_img.shape[1])
    #cv2.imshow('gray', box_img)
    offset2 = math.ceil(box_img.shape[1] / 3)
    list_circle = []
    offset = 30
    list_answers = []
    for j in range(3):
        list_answers.append(box_img[:,j * offset2:(j + 1) * offset2])
        for i in range(10):
            bubble_choice = list_answers[j][i * offset: (i + 1) * offset,: ]
            bubble_choice = cv2.threshold(bubble_choice, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            bubble_choice = cv2.resize(bubble_choice, (28, 28), cv2.INTER_AREA)
            bubble_choice = bubble_choice.reshape((28, 28, 1))
            list_circle.append(bubble_choice)
            #cv2.imwrite(f'result/id{j}{i}.png',bubble_choice )
    # print(len(list_circle))
    return list_circle



def finCircle_idstudent(imgrect):
    #result = []
    gray = cv2.cvtColor(imgrect, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray', gray)
    box_img = gray[3:gray.shape[0] - 3, 7:gray.shape[1]-10]
    box_img = ResizeImage(box_img, height=400)
    #print(box_img.shape[0], box_img.shape[1])
    #cv2.imshow('gray', box_img)
    offset2 = math.ceil(box_img.shape[1] / 6)
    list_circle = []
    offset = 40
    list_answers = []
    for j in range(6):
        list_answers.append(box_img[:,j * offset2:(j + 1) * offset2])
        for i in range(10):
            bubble_choice = list_answers[j][i * offset: (i + 1) * offset,: ]
            bubble_choice = cv2.threshold(bubble_choice, 50, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            bubble_choice = cv2.resize(bubble_choice, (28, 28), cv2.INTER_AREA)
            bubble_choice = bubble_choice.reshape((28, 28, 1))
            list_circle.append(bubble_choice)
            cv2.imwrite(f'result/idstudent{j}{i}.png',bubble_choice )


    # print(len(list_circle))
    return list_circle





def map_answer(idx):
    if idx % 4 == 0:
        answer_circle = "A"
    elif idx % 4 == 1:
        answer_circle = "B"
    elif idx % 4 == 2:
        answer_circle = "C"
    else:
        answer_circle = "D"
    return answer_circle

def map_answer_ds(idx):
    if idx % 2 == 0:  # chia lay du, tim so chan va le
        answer_circle = "D"
    elif idx % 2 == 1:
        answer_circle = "S"
    return answer_circle




def get_answers(list_ans):
   results = defaultdict(list)
   #model = CNN_Model('weight.h5').build_model(rt=True)
   list_ans = np.array(list_ans)
   scores = model.predict_on_batch(list_ans / 255.0)
   for idx, score in enumerate(scores):
       question = idx // 4
       # score [unchoiced_cf, choiced_cf]
       if score[1] > 0.8:      # choiced confidence score > 0.9
           chosed_answer = map_answer(idx)
           results[question+1].append(chosed_answer)
       #else:
           #results[question + 1].append("N")
   return results

def get_answers_idtest(list_ans):
   results = defaultdict(list)
   #model = CNN_Model('weight.h5').build_model(rt=True)
   list_ans = np.array(list_ans)
   scores = model.predict_on_batch(list_ans / 255.0)
   myVal = np.zeros((10, 3))
   countC = 0
   countR = 0
   for value in scores:
       if value[1] > 0.7:
           myVal[countR, countC] = 1
       else:
           myVal[countR, countC] = -1
       countR += 1
       if countR == 10:
           countC += 1
           countR = 0
   #print(myVal)
   ketqua = ""
   for cols in range(0, 3):
       for rows in range(0, 10):
           arr = myVal[rows][cols]
           if arr == 1:
               ketqua = ketqua + str(rows)

   return ketqua

def get_answers_idstudent(list_ans):
   list_ans = np.array(list_ans)
   scores = model.predict_on_batch(list_ans / 255.0)
   myVal = np.zeros((10, 6))
   countC = 0
   countR = 0
   for value in scores:
       if value[1] > 0.7:
           myVal[countR, countC] = 1
       else:
           myVal[countR, countC] = -1
       countR += 1
       if countR == 10:
           countC += 1
           countR = 0
   #print(myVal)
   ketqua = ""
   for cols in range(0, 6):
       for rows in range(0, 10):
           arr = myVal[rows][cols]
           if arr == 1:
               ketqua = ketqua + str(rows)

   return ketqua




def get_answers_ds(list_ans):
   results = defaultdict(list)
   #model = CNN_Model('weight.h5').build_model(rt=True)
   list_ans = np.array(list_ans)
   scores = model.predict_on_batch(list_ans / 255.0)
   for idx, score in enumerate(scores):
       question = idx // 8
       # score [unchoiced_cf, choiced_cf]
       if score[1] > 0.7:      # choiced confidence score > 0.9
           chosed_answer = map_answer_ds(idx)
           results[question+1].append(chosed_answer)


   return results


def get_answers_tl(list_ans):
   results = ""
   list_ans = np.array(list_ans)
   scores = model.predict_on_batch(list_ans / 255.0)
   dapan=""
   phay=""
   for idx, score in enumerate(scores):
       if score[1] > 0.9:# choiced confidence score > 0.9
           if idx ==0:
               dapan = "-"
           elif idx ==1:
               phay= 'chuc'
           elif idx ==2:
               phay = "tram"
       if idx==3:
           break
   #print(phay)
   list_so = scores[3:]
   myVal = np.zeros((10, 4))
   countC = 0
   countR = 0
   for value in list_so:
       if value[1]>0.9:
           myVal[countR, countC] = 1
       else:
           myVal[countR, countC] = -1
       countC += 1
       if countC == 4:
           countR += 1
           countC = 0
   #print(myVal)
   ketqua = ""
   for cols in range(0, 4):
       for rows in range(0, 10):
           arr = myVal[rows][cols]
           if arr == 1:
               ketqua = ketqua + str(rows)
   if dapan =="-":
       #co hai truong hop co phay va khong
       if phay=="tram":
           ketqua = ketqua[:1] + "," + ketqua[1:]
       ketqua = "-"+ketqua
   else:
       if phay == "chuc":
           ketqua = ketqua[:1] + "," + ketqua[1:]
       if phay == "tram":
           ketqua = ketqua[:2] + "," + ketqua[2:]
       else:
           ketqua = ketqua


   return ketqua



def preprocess_image(img_array):
    img_array = img_array / 255.0  # Chuẩn hóa giá trị pixel về [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Thêm batch dimension
    return img_array



#get theo diem anh
def get_answers_by_pixcel(list_ans):
    mypixcelVal = np.zeros((40, 4))
    countC = 0
    countR = 0
    for image in list_ans:
        totalPixcel = cv2.countNonZero(image)
        mypixcelVal[countR,countC] = totalPixcel
        countC +=1
        if countC == 4:
            countR +=1
            countC =0

    myIndex = []
    print(mypixcelVal)

def drawCircleAnswer(chieurong, anchor, liscircle):
    print(anchor[1])
    vitri = anchor[0]-chieurong
    offset = 15
    print(vitri)
    for idx in range(9):
        cv2.circle(img, (int(vitri)+25, int(anchor[1]+20)+(idx*offset)), 5, (0, 0, 255), 1)




if __name__ == '__main__':
    model = CNN_Model('weight.h5').build_model(rt=True)
    img = cv2.imread('images/test2.jpg')
    print("chieu rong", img.shape[1], "chieu cao", img.shape[0])
    #CNN_Model().train()
    anchors = []
    # tim toa do hinh vuong lon trong hinh, tu toa do tinh ra tam cua hinh vuong
    anchor = timhinh2(img)

    anchors.append(anchor)
    anchors.append((anchor[0] - 4, anchor[1] + 41))
    # hai diem ben sbd va de
    anchors.append((anchor[0] + 90, anchor[1] - 17))
    anchors.append((anchor[0] + 90, anchor[1] - 17 - 86))
    # phan cau tra loi
    anchors.append((anchor[0] - 4 - 132, anchor[1] + 41))
    anchors.append((anchor[0] - 4 - (2 * 132), anchor[1] + 41))
    anchors.append((anchor[0] - 4, anchor[1] + 41 + 158))
    anchors.append((anchor[0] - 4 - 132, anchor[1] + 41 + 158))
    anchors.append((anchor[0] - 4 - (132 * 2), anchor[1] + 41 + 158))

    anchors.append((anchor[0] - 4, anchor[1] + 41 + 158 + 23))
    anchors.append((anchor[0] - 4 - 132, anchor[1] + 41 + 158 + 23))
    anchors.append((anchor[0] - 4 - (2 * 132), anchor[1] + 41 + 158 + 23))

    anchors.append((anchor[0] - 4, anchor[1] + 41 + 158 + 23 + 92))
    anchors.append((anchor[0] - 4 - 132, anchor[1] + 41 + 158 + 23 + 92))
    anchors.append((anchor[0] - 4 - (132 * 2), anchor[1] + 41 + 158 + 23 + 92))

    anchors.append((anchor[0] + 32, anchor[1] + 41 + 158 + 23 + 103))
    anchors.append((anchor[0] + 32 - 84, anchor[1] + 41 + 158 + 23 + 103))
    anchors.append((anchor[0] + 32 - (84 * 3), anchor[1] + 41 + 158 + 23 + 103))
    anchors.append((anchor[0] + 32 - (84 * 4), anchor[1] + 41 + 158 + 23 + 103))

    anchors.append((anchor[0] + 32, anchor[1] + 41 + 158 + 23 + 103 + 238))
    anchors.append((anchor[0] + 32 - 84, anchor[1] + 41 + 158 + 23 + 103 + 238))
    anchors.append((anchor[0] + 32 - (84 * 2), anchor[1] + 41 + 158 + 23 + 103 + 238))
    anchors.append((anchor[0] + 32 - (84 * 3), anchor[1] + 41 + 158 + 23 + 103 + 238))
    anchors.append((anchor[0] + 32 - (84 * 4), anchor[1] + 41 + 158 + 23 + 103 + 238))
    listPoint = ClusterPoints(anchors, 3)
    # sort tu tren xuong, tu phai qua trai
    sorted_points = sorted(listPoint, key=lambda x: (x[1]))  # 1 sort theo truc y
    anchors = sorted_points;
    print('anchor')
    print(anchors)
    adjusted_coordinates = adjust_coordinates(anchors)

    list_new = []
    for findcent in adjusted_coordinates:
        cne = timhinh3(img, findcent)
        list_new.append(cne)
    print(f'new                  {list_new}')
    for idx, center in enumerate(list_new):
        if idx ==23:
            print(center)
            cv2.circle(img, (center[0], center[1]), 1, (0, 0, 255), -1)
    if len(list_new) == 24:
        # for cnt in sorted_coordinates:
        # cv2.rectangle(img, (cnt), (cnt[0] + 1, cnt[1] + 1), (0, 0, 255), 2)

        anchors = TranformPoints(list_new, (0, 0), img.shape[0] / 810.0)
        # phan xu li id va sbd
        chieucao = anchors[1][1] - anchors[0][1]
        idtest = [(anchors[0][0], anchors[0][1] - chieucao), (img.shape[0], anchors[1][1])]
        idtest = sub_rect_image(img, idtest)
        list_circle_result = finCircle_idtest(idtest)
        result = get_answers_idtest(list_circle_result)
        print(f'idtest: {result}')
        cv2.putText(img, result, (int(anchors[0][0]), int(anchors[0][1] - chieucao)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1, cv2.LINE_AA)
        # idstudent
        idstudent = [(anchors[2][0], anchors[0][1] - chieucao), anchors[1]]
        idstudent = sub_rect_image(img, idstudent)
        list_circle_result = finCircle_idstudent(idstudent)
        result = get_answers_idstudent(list_circle_result)
        cv2.putText(img, result, (int(anchors[2][0]), int(anchors[0][1] - chieucao)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1, cv2.LINE_AA)
        print(f'idstudent: {result}')
        # phan I
        box_answer = []
        chieurong = anchors[3][0] - anchors[4][0]
        kv1 = [(anchors[5][0] - chieurong, anchors[5][1]), anchors[8]]
        kv2 = [anchors[5], anchors[7]]
        kv3 = [anchors[4], anchors[6]]
        kv4 = [anchors[3], (anchors[6][0] + chieurong, anchors[6][1])]
        img_kv1 = sub_rect_image(img, kv1)
        box_answer.append(img_kv1)
        img_kv2 = sub_rect_image(img, kv2)
        box_answer.append(img_kv2)
        img_kv3 = sub_rect_image(img, kv3)
        box_answer.append(img_kv3)
        img_kv4 = sub_rect_image(img, kv4)
        box_answer.append(img_kv4)
        list_circle = finCircle(box_answer)
        if len(list_circle)==160:
            drawCircleAnswer(chieurong, anchors[5], list_circle)

        #for idx, imgt in enumerate(box_answer):
            #cv2.imshow(f'result/20_1_{idx}.png', imgt)

        #for idx, imgt in enumerate(list_circle):
            #cv2.imwrite(f'result/21_1_{idx}.png', imgt)



        ''' 
        for idx, imgt in enumerate(list_circle):
            cv2.imwrite(f'result/20_1_{idx}.png', imgt)
        phan nay dung de lay ra cac hinh nham muc dich train data
        cau = 0
        for idx, boxtem in enumerate(list_circle):
            cau = cau + 1;
            if (idx + 1) % 4 == 0:
                cau = 0
            ques = idx // 4
            # cv2.imwrite(f'result/a{ques+1}{cau}.png',boxtem)
        '''
        result = get_answers(list_circle)
        print(result)

        # phan II trac nghiem dung sai #
        list_img_ds = []
        ds1 = [(anchors[11][0] - chieurong, anchors[11][1]), anchors[14]]
        ds1 = sub_rect_image(img, ds1)
        list_img_ds.append(ds1)
        ds2 = [anchors[11], anchors[13]]
        ds2 = sub_rect_image(img, ds2)
        list_img_ds.append(ds2)
        ds3 = [anchors[10], anchors[12]]
        ds3 = sub_rect_image(img, ds3)
        list_img_ds.append(ds3)
        ds4 = [anchors[9], (anchors[12][0] + chieurong, anchors[12][1])]
        ds4 = sub_rect_image(img, ds4)
        list_img_ds.append(ds4)
        list_circle_result = []

        for imgitem in list_img_ds:
            list_circle_result.extend(finCircle_ds(imgitem))

        if len(list_circle_result) == 64:
            result = get_answers_ds(list_circle_result)
            print(result)

            # phan III dien so #
        list_img_tl = []
        chieurong = anchors[15][0] - anchors[16][0]
        tl1 = [(anchors[18][0] - chieurong, anchors[18][1]), anchors[23]]
        tl1 = sub_rect_image(img, tl1)
        #cv2.imshow('tl1', tl1)
        tl3 = [anchors[17], anchors[21]]
        tl3 = sub_rect_image(img, tl3)
        tl4 = [(anchors[16][0] - chieurong, anchors[16][1]), anchors[20]]
        tl4 = sub_rect_image(img, tl4)
        tl5 = [anchors[16], anchors[19]]
        tl5 = sub_rect_image(img, tl5)
        tl2 = [anchors[18], anchors[22]]
        tl2 = sub_rect_image(img, tl2)
        #cv2.imshow('tl2', tl2)
        tl6 = [anchors[15], (anchors[19][0] + chieurong, anchors[19][1])]
        tl6 = sub_rect_image(img, tl6)
        list_img_tl.append(tl1)
        list_img_tl.append(tl2)
        list_img_tl.append(tl3)
        list_img_tl.append(tl4)
        list_img_tl.append(tl5)
        list_img_tl.append(tl6)
        result_tl = defaultdict(list)
        for idx, imgt in enumerate(list_img_tl):
           #cv2.imshow(f'{idx}', imgt)
           list_circle_result = finCircle_tl(imgt)
           result = get_answers_tl(list_circle_result)
           result_tl[idx + 1] = result

        print(result_tl)
        offset = 0
        for value in result_tl.values():
            cv2.putText(img, value,
                        (int(anchors[18][0] - chieurong + 30 + (offset * chieurong)), int(anchors[18][1] + 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1, cv2.LINE_AA)
            offset = offset + 1



    #vi tim da theo thu tu nen khong can sort
    '''
    grouped_by_y = defaultdict(list)
    for x, y in list_new:
        grouped_by_y[y].append((x, y))
    # Sắp xếp các nhóm theo x từ lớn đến nhỏ
    sorted_coordinates = []
    for y in sorted(grouped_by_y.keys()):
        sorted_group = sorted(grouped_by_y[y], key=lambda coord: coord[0], reverse=True)
        sorted_coordinates.extend(sorted_group)
    print(f'sorted_coordinates {sorted_coordinates}')
    '''


    #for ixd, imgtem in enumerate(list_circle_result):
            #cv2.imwrite(f'result/tl_{ixd}.png', imgtem)











    '''
        list1 = finCircle_ds(ds1)
        for ixd, imgtem in enumerate(list1):
            cv2.imshow(f'{ixd}', imgtem)

        result = get_answers_ds(list1)
        print(result)
    '''





        #cv2.imshow('ds', ds4)
































    #cv2.drawContours(img, [anchors[0]], 0, (0, 255, 0), 2)
    #cv2.drawContours(img, [anchors[1]], 0, (0, 255, 0), 2)


    #for anchor in anchors:
        #cv2.drawContours(img, [anchor], 0, (0, 255, 0), 2)

    cv2.imshow('hieu', img)
    ##for blockimg in cropImage:
       ## ans_block_img = np.array(blockimg[0])
        ##cv2.imshow('h', ans_block_img)

    #img = ResizeImage(img, height=800)
    #cv2.imshow('hieu', img)
    #cv2.imwrite('result/'+str(uuid.uuid4())+'.jpg', img)

    cv2.waitKey()


