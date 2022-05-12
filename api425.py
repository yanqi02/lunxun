import datetime

import cv2
import numpy as np
import requests
import torch
from flask import Flask
# -*- coding: UTF-8 -*-
import os
import json
from flask_cors import *
from loguru import logger
import loguru
from img2Base64 import str2img, img2str
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import non_max_suppression, check_img_size, scale_coords
from utils.plots import plot_one_box

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
from flask import Flask, request
import base64
import threading
import time


def getImgByRtsp(rtsp):
    cap = cv2.VideoCapture(rtsp)
    ret, img = cap.read()
    readmun = 0
    success = False
    while True:
        try:
            readmun += 1
            # print(self.readmun)
            ret, img = cap.read()
            if (img is not None):
                success = True
                break
            if readmun >= 50:
                print("not get!")
                break
        except:
            pass
    cap.release()
    if success:
        return img
    else:
        return None


# def mythread(info):
#     print(info)
#     # areaId = info["areaId"]
#     equipmentId = info["equipmentId"]
#     # eqName = info["equipmentName"]
#     imgId=1
#     time=str(datetime.datetime.now())
#     rtsp = info['rtsp']
#     im0, isViolate,weiguilist = detect_one1(rtsp,model1=model1,model2=model2)
#     print(isViolate,weiguilist)
#     isViolate=True
#     if im0 is not None:
#         cv2.imshow("aa", im0)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     if (im0 is not None) and isViolate==True:
#         imgBase = img2str(im0)
#
#         eventType="1111"
#         typeText="zzzz"
#         result = {'areaId': "as", "equipmentId": equipmentId,
#                   "equipmentName":"SAS","eventTime":time,"eventType":eventType,
#                   "eventTypeTxt":typeText,"imgBase":imgBase,"imgId":imgId,"monitorArea":"aa","redPoint":[1,1,1,1]}
#         try:
#                 r = requests.post("http://10.67.206.234:30172/api/sys/abnormal/event/ShiyouDaxueReceive", json=result)
#                 print("sDDasdasd:",r.json())
#         except Exception as e:
#                 print("except",e)
# if im0 is not None:
#     cv2.imshow("aa", im0)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# if (im0 is not None) and isViolate==True:
#     imgBase = img2str(im0)
#     for wz in weiguilist:
#         eventType=wz
#         typeText="as"
#         result = {'areaId': "as", "equipmentId": equipmentId,
#               "equipmentName":"SAS","eventTime":time,"eventType":eventType,
#               "eventTypeTxt":typeText,"imgBase":imgBase,"imgId":imgId,"monitorArea":"aa","redPoint":[1,1,1,1]}
#         try:
#             r = requests.post("http://10.67.206.234:30172/api/sys/abnormal/event/ShiyouDaxueReceive", json=result)
#             print(r.json())
#         except Exception as e:
#             print(e)

def mythread(info):
    logger.debug(info)
    print(info)
    # areaId = info["areaId"]
    equipmentId = info["equipmentId"]
    # eqName = info["equipmentName"]
    imgId = 1
    time = str(datetime.datetime.now())
    rtsp = info['rtsp']
    im0, isViolate, weiguilist = detect_one1(rtsp, model1=model1, model2=model2, area=equipmentId)
    print(isViolate, weiguilist)
    # isViolate=True

    # if im0 is not None:
    #     cv2.imshow("aa", im0)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    """{
  "areaId": "00108300412169490891",
  "equipmentId": "1b1cfe27ac9e46c7be53688581a41831",
  "equipmentName": "1",
  "eventTime": "2022-03-22 07:48:15.598",
  "eventType": "101",
  "eventTypeTxt": "phone",
  "imgBase": "iVBORw0KGgoAAAANSUhEUgAAADIAAAAUCAIAAABAqPnNAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAABnSURBVEhL7ZUxDgAgCAPB//8ZjRriaNUaBm7RjQuEomYm8SjzDUZqIaQWArCJqjp/HeoKnwfEsCTJ3eYWSe5NnD6fLyXl71tI0Rp4Cw9KELWcdcSb5X5oOc0votY+eXwQUgshpJZIBe6mNgFrJ31kAAAAAElFTkSuQmCC",
  "imgId": "1",
  "monitorArea": "1",
  "redPoint": "[1.0,1.0,2.0,2.0]"
}"""

    if (im0 is not None) and isViolate == True:
        imgBase = img2str(im0)
        for wz in weiguilist:
            eventType = wz
            typeText = weiguilist[wz]
            result = {'areaId': "00108300412169490891", "equipmentId": equipmentId,
                      "equipmentName": "1", "eventTime": time, "eventType": eventType,
                      "eventTypeTxt": typeText, "imgBase": imgBase, "imgId": imgId, "monitorArea": "aa",
                      "redPoint": "[1.0,1.0,1.0,1.0]"}
            try:
                r = requests.post("http://10.67.206.234:30172/api/sys/abnormal/event/ShiyouDaxueReceive", json=result)
                print("result:", result)
                print("Result:", r.json())
            except Exception as e:
                print("Exception:", e)


app = Flask(__name__)


def detectOne(path):
    weights = "./newbest.pt"
    weights2 = "./1115erci.pt"
    device = torch.device('cuda', 0)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    model2 = attempt_load(weights2, map_location=device)  # load FP32 model
    half = True
    model.half()  # to FP16
    model2.half()
    # img = cv2.imread(path)
    img = getImgByRtsp(rtsp=path)
    if img is None:
        return None, None, None
    img_size = 640
    im0 = img
    stride, names = int(model.stride.max()), model.names
    img_size = check_img_size(img_size, s=stride)
    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()  # to FP16
    img = letterbox(im0, auto=True, new_shape=img_size, stride=32)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    # model.warmup(imgsz=(1, 3, *img_size), half=half)  # warmup
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        # img = img[None]  # expand for batch dim
        img = img.unsqueeze(0)
    pred = model(img, augment=True)[0]
    pred = non_max_suppression(pred)
    for i, det in enumerate(pred):
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            info_list = []
            for *xyxy, conf, cls in reversed(det):
                xyxy = torch.tensor(xyxy).view(-1).tolist()
                info = [xyxy[0], xyxy[1], xyxy[2], xyxy[3], int(cls)]
                info_list.append(info)
                im0 = plot_one_box(xyxy, im0, label=str(cls), color=[255, 0, 0], line_thickness=3)

            print(info_list)

        else:
            print("NOOOO")
            return im0, None, True
    isViolate = True
    return im0, info_list, isViolate


class loadmodel:
    def __init__(self, weights, device):
        self.weights = weights
        self.device = device
        self.model = attempt_load(weights, map_location=device)  # load FP32 model
        self.model.half()  # to FP16

    def getModel(self):
        return self.model


from util.util import *
from modules.load_state import load_state
import psutil
import os
from models.with_mobilenet import PoseEstimationWithMobileNet
from utils.datasets import LoadStreams3
from loguru import logger
from RTSP.YunTai import *

logger.debug('this is a debug message')

global PhoneList, SmokeList, PersonList, QZList, PTList, ClothList, NoSafeHatList, GuardianList, OilList, LineList, PTpersoncrossList, QZpersoncrossList, CaiYouShuList
distense_n = 3
TF_noclothes = False
TF_QZpeoplecross = False
TF_PTpeoplecross = False
TF_QZ = False
TF_YHopration = False
TF_YHNOxfq = False
TF_YHYQPwofang = False
TF_YHO2Pwofang = False
TF_YHTongCheYunShu = False
TF_YHJDQP = False
TF_JJBZ = False
TF_PT = False
TF_NOG = False
TF_phone = False
TF_smoke = False
TF_NoSafeHat = False
TF_wuran = False
TF_noline = False
TF_PersonCaiyoushu = False
TF_noline = False
TF_Pipline = False
TF_beltnomask = False
TF_pangenoil = False
TF_piplineoil = False
TF_ketoujistop = False
TF_GHGQpeoplegather = False  # 高后果区人员聚集
TF_ArmPerson = False
TF_perInarm = False

PhoneList = []
SmokeList = []
PersonList = []
YiQueList = []
YangQiList = []
XiaoFangList = []
QZList = []
PTList = []
YHList = []
YQPList = []
YQPDOWNList = []
O2PList = []
YHtljList = []
QPDOWNList = []
O2PDOWNList = []
XFList = []
KEYPointsList = []
ClothList = []
NoSafeHatList = []
GuardianList = []
OilList = []
LineList = []
PTpersoncrossList = []
QZpersoncrossList = []
CaiYouShuList = []
KetoujiheadList = []
KetoujiList = []
PiplineList = []
BeltList = []
QZarmList = []

peopleID = 0
YH_whiteID = 1
YH_blueID = 2
YH_redID = 3
YH_tljID = 4
PT_WaJueJiID = 5
flagID = 6
oilID = 7
caiyoushuID = 8
ketouji_headID = 10
ketouji_armID = 11
ketouji_bottomID = 12
ketoujiID = 9
chouyoujiID = 13
QZ_armID = 14
QZ_SuiCheDiaoID = 15
QZ_QiDiaoID = 16
YH_DHJID = 17

# 采油二次类号：
PhoneID = 0
SmokeID = 1
No_SafeHatID = 3
SafeHatID = 2
GuardianID = 4
maskID = 5
no_maskID = 6
HumujingID = 7
No_HumujingID = 8
SafeBeltID = 9
WinSafeHatID = 10

TF_save = [False] * 10
TF_save_data = [False] * 10
save_video_path = 'save_video/'

draw_dic = {}
person_imgg = []
draw_flg = True
flg = True
second_img = False
pic_num = 0
color_dict = colorList.getColorList()
save_num = 0
stop = 0

contours = []
weizhi_list = []
stop_list = [0] * 10
weizhi = [0] * 4
temp_weizhi = [0] * 4
frame_lwpCV_list = []
frame_lwpCV0_list = []
area_list = []
rect_area_list = []
gray_lwpCV_list = []
pre_frame_list = [0] * 10
flag_dic = {}
stop_dic = []
p = 0
sum = 0


def automatedMSRCR(img, sigma_list):
    img = np.float64(img) + 1.0

    img_retinex = multiScaleRetinex(img, sigma_list)

    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break

        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break

        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)

        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255

    img_retinex = np.uint8(img_retinex)

    return img_retinex


# 一次检测
def detect_one(path):
    weights = "./newbest.pt"
    weights2 = "./1115erci.pt"
    device = torch.device('cuda', 0)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    model2 = attempt_load(weights2, map_location=device)  # load FP32 model
    half = True
    model.half()  # to FP16
    model2.half()
    img = cv2.imread(path)
    # img = getImgByRtsp(rtsp=path)
    if img is None:
        return None, None, None
    img_size = 640
    im0 = img
    stride, names = int(model.stride.max()), model.names
    img_size = check_img_size(img_size, s=stride)
    img = letterbox(im0, auto=True, new_shape=img_size, stride=32)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    # model.warmup(imgsz=(1, 3, *img_size), half=half)  # warmup
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        # img = img[None]  # expand for batch dim
        img = img.unsqueeze(0)
    pred = model(img, augment=True)[0]
    pred = non_max_suppression(pred)
    for i in range(10):
        TF_save[i] = False
    for each in os.listdir('./ztry'):
        os.remove('./ztry/' + each)

    if half:
        model.half()  # to FP16
        model2.half()

    view_img = True
    cudnn.benchmark = True  # set True to speed up constant image size inference

    # Get names and colors
    # 获取类别名字
    names = model.module.names if hasattr(model, 'module') else model.names
    print(names)
    # 设置画框的颜色
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    cate_ce = {}
    dic = {}
    pp = 0
    global person_imgg, flg, second_img

    for i, det in enumerate(pred):  # detections per image
        PhoneList.clear()
        SmokeList.clear()
        PersonList.clear()
        QZList.clear()
        PTList.clear()
        ClothList.clear()
        NoSafeHatList.clear()
        GuardianList.clear()
        OilList.clear()
        LineList.clear()
        PTpersoncrossList.clear()
        QZpersoncrossList.clear()
        CaiYouShuList.clear()
        draw_dic.clear()
        YHList = []
        KetoujiheadList = []
        KetoujiList = []
        YQPList = []
        O2PList = []
        XFList = []
        YHtljList = []
        QZarmList.clear()

        TF_noclothes = False
        TF_QZ = False
        TF_YHopration = False
        TF_PT = False
        TF_NOG = False
        TF_phone = False
        TF_smoke = False
        TF_NoSafeHat = False
        TF_wuran = False
        TF_weigui = False
        TF_perInarm = False

        # 如果输入源是webcam，则batch_size不为1，取出dataset中的一张图片
        dis_time = datetime.datetime.now()

        yuantu = im0.copy()
        # 设置打印信息(图片长宽)
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            # 调整预测框的坐标：基于resize+pad的图片的坐标-->基于原size图片的坐标
            # 此时坐标格式为xyxy
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            # 保存预测结果
            for *xyxy, conf, cls in reversed(det):

                # 在原图上画框
                if True:  # Add bbox to image
                    c = int(cls)  # integer class
                    # if (c != PhoneID and c != SmokeID):
                    label = f'{names[c]} {conf:.2f}'
                    labelx = label
                    label = label.split(' ')[0]
                    yuzhi = labelx.split(' ')[1]
                    yuzhi = float(yuzhi)

                    # im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                    if (c == peopleID and yuzhi >= 0.9):
                        PersonList.append(xyxy)
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)

                    elif (c == QZ_SuiCheDiaoID and yuzhi >= 0.85):
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                        QZList.append(xyxy)
                    elif (c == QZ_QiDiaoID and yuzhi >= 0.85):
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                        QZList.append(xyxy)

                    elif (c == YH_blueID and yuzhi >= 0.6):
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                        O2PList.append(xyxy)  # 氧气瓶
                        YHList.append(xyxy)
                    elif (c == YH_whiteID and yuzhi >= 0.6):
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                        YQPList.append(xyxy)  # 乙炔瓶
                        YHList.append(xyxy)

                    elif (c == YH_redID and yuzhi >= 0.4):
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                        XFList.append(xyxy)  # 灭火器


                    elif (c == QZ_armID and yuzhi >= 0.7):
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                        QZarmList.append(xyxy)
                    elif (c == YH_tljID and yuzhi >= 0.4):
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                        YHtljList.append(xyxy)
                    elif (c == PT_WaJueJiID and yuzhi >= 0.7):
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                        PTList.append(xyxy)

                    elif (c == ketouji_headID and yuzhi >= 0.65):
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                        KetoujiheadList.append(xyxy)
                    elif (c == ketoujiID and yuzhi >= 0.65):
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                        KetoujiList.append(xyxy)

                    elif (c == oilID and yuzhi >= 0.7):
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                        OilList.append(xyxy)

                    elif (c == flagID and yuzhi >= 0.25):
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                        LineList.append(xyxy)

        new_img = im0.copy()
        # 二次检测
        # 6 动土作业
        if PersonList and PTList:
            TF_PT = True
            # 15溢油
        if OilList:
            for oil in OilList:
                oilx1 = int(oil[0].item())  # x1
                oily1 = int(oil[1].item())  # y1
                oilx2 = int(oil[2].item())  # x2
                oily2 = int(oil[3].item())  # y2
                oilxmid = (oilx1 + oilx2) // 2
                oilymid = (oily1 + oily2) // 2
                oilimg = im0.copy()

                oil_part = []
                oil_part.append(oilimg[oily1:oilymid, oilx1:oilxmid])
                oil_part.append(oilimg[oily1:oilymid, oilxmid:oilx2])
                oil_part.append(oilimg[oilymid:oily2, oilx1:oilxmid])
                oil_part.append(oilimg[oilymid:oily2, oilxmid:oilx2])
                black_time = 0
                blue_time = 0
                for part in oil_part:
                    oil_color = get_color(part)
                    print('oil color : ', oil_color)
                    if oil_color == 'blue' or 'black':
                        black_time = black_time + 1
                        blue_time = blue_time + 1
                    # print('blacktime:---------',black_time)
                    if (black_time >= 2 or blue_time >= 3):
                        TF_wuran = True

        # 5 起重作业
        if QZList and PersonList:
            TF_QZ = True
        nosafehat2 = False
        # print(PersonList)
        if PersonList:
            num = 0

            for person in PersonList:
                personx1 = person[0]
                persony1 = person[1]
                personx2 = person[2]
                persony2 = person[3]

                x1 = int(personx1.item())
                y1 = int(persony1.item())
                x2 = int(personx2.item())
                y2 = int(persony2.item())
                person_img = im0[y1: y2, x1: x2]
                cv2.imwrite('./try/try' + str(i) + '/{}.jpg'.format(num), person_img)
                source2 = './try/try' + str(i)
                dataset2 = LoadImages(source2)
                names2 = model2.module.names if hasattr(model2, 'module') else model2.names

                # 设置画框的颜色
                colors2 = [0, 0, 255]
                for path2, img_2, im0s_2, vid_cap in dataset2:
                    if img_2 is None:
                        continue
                    img_2 = torch.from_numpy(img_2).to(device)
                    # 图片也设置为Float16
                    img_2 = img_2.half() if half else img_2.float()  # uint8 to fp16/32
                    img_2 /= 255.0  # 0 - 255 to 0.0 - 1.0
                    # 没有batch_size的话则在最前面添加一个轴
                    if img_2.ndimension() == 3:
                        img_2 = img_2.unsqueeze(0)

                    pred2 = model2(img_2, augment=True)[0]

                    pred2 = non_max_suppression(pred2)

                    im0_2 = im0s_2.copy()
                    for i2, det2 in enumerate(pred2):  # detections per image
                        # 如果输入源是webcam，则batch_size不为1，取出dataset中的一张图片

                        p2, s2, im0_2, frame = path2, '', im0s_2.copy(), getattr(dataset2, 'frame', 0)
                        if len(det2):
                            det2[:, :4] = scale_coords(img_2.shape[2:], det2[:, :4], im0_2.shape).round()
                            for *xyxy2, conf, cls2 in reversed(det2):
                                if True:  # Add bbox to image
                                    c2 = int(cls2)  # integer clas

                                    label = f'{names2[c2]} {conf:.2f}'
                                    labelx = label
                                    label = label.split(' ')[0]
                                    yuzhi = labelx.split(' ')[1]
                                    yuzhi = float(yuzhi)
                                    if (c2 == GuardianID):
                                        im0_2 = plot_one_box(xyxy2, im0_2, label=label, color=colors2, line_thickness=3)
                                        GuardianList.append(xyxy2)

                                    if c2 == No_SafeHatID and yuzhi >= 0.6:
                                        TF_NoSafeHat = True
                                        im0_2 = plot_one_box(xyxy2, im0_2, label=label, color=colors2, line_thickness=3)

                                    if (c2 == PhoneID and yuzhi >= 0.1):  # 3. 违禁使用手机
                                        im0_2 = plot_one_box(xyxy2, im0_2, label=label, color=colors2,
                                                             line_thickness=3)

                                        PhoneList.append(xyxy2)
                                        TF_phone = True
                                    if (c2 == SmokeID and yuzhi >= 0.1):  # 2 生产区域吸烟
                                        im0_2 = plot_one_box(xyxy2, im0_2, label=label, color=colors2,
                                                             line_thickness=3)
                                        SmokeList.append(xyxy2)
                                        TF_smoke = True

                    im0[y1:y2, x1:x2] = im0_2
                    new_img[y1:y2, x1:x2] = im0_2

            # 2.工衣判断
            logger.debug("工衣判断")

            now_time = datetime.datetime.now().hour
            if time.tzname[0] == 'UTC':
                now_time = (now_time + 8) % 24
            if 5 <= now_time <= 19:
                for person in PersonList:
                    imagea = im0.copy()
                    px1 = person[0]
                    py1 = person[1]
                    px2 = person[2]
                    py2 = person[3]
                    personxmin = int(px1.item()) + 5  # 获取人的图片和坐标
                    personymin = int(py1.item()) + 5
                    personxmax = int(px2.item()) - 5
                    personymax = int(py2.item()) - 5
                    if (personxmax - personxmin) < 10 or (personymax - personymin) < 20:
                        continue
                    personmidx1 = personxmin + (personxmax - personxmin) // 3
                    personmidx2 = personxmax - (personxmax - personxmin) // 3
                    personmidy = (personymin + personymax) // 2
                    person_part = []
                    # person_part.append(imagea[personmidy - personmidy//2:personmidy + personmidy//2,personmidx - personmidx//2:personmidx + personmidx//2])
                    person_part.append(imagea[personymin:personmidy, personxmin:personmidx1])
                    person_part.append(imagea[personymin:personmidy, personmidx1:personmidx2])
                    person_part.append(imagea[personymin:personmidy, personmidx2:personxmax])

                    ff = 0
                    for each_part in person_part:

                        clothes_color = get_color(each_part)
                        logger.info(clothes_color)
                        # print('********color**********', clothes_color)
                        if clothes_color == 'red' or clothes_color == 'red2':
                            ff = ff + 1
                            break
                    person_part.clear()
                    person_part.append(imagea[personmidy:personymax, personxmin:personmidx1])
                    person_part.append(imagea[personmidy:personymax, personmidx1:personmidx2])
                    person_part.append(imagea[personmidy:personymax, personmidx2:personxmax])
                    for each_part in person_part:
                        clothes_color = get_color(each_part)
                        logger.debug(clothes_color)

                        if clothes_color == 'red' or clothes_color == 'red2':
                            ff = ff + 1
                            break
                    person_part.clear()
                    if ff != 2:
                        # draw_dic.setdefault("person",[]).append(person)
                        TF_noclothes = True
                    if ff == 1 and (personxmax - personxmin) / (personymax - personymin) > 0.45:
                        TF_noclothes = False

                    logger.error(len(PersonList))
                    logger.error("未穿工衣：" + str(TF_noclothes) + '   ' + str(ff))
                    if TF_noclothes == True:
                        break

        # 13 起重作业人员未离开
        if ((QZarmList) and (PersonList)):
            for QZ in QZarmList:
                QZx1 = int(QZ[0].item())
                QZy1 = int(QZ[1].item())
                QZx2 = int(QZ[2].item())
                QZy2 = int(QZ[3].item())
                for per in PersonList:
                    perx1 = int(per[0].item())
                    pery1 = int(per[1].item())
                    perx2 = int(per[2].item())
                    pery2 = int(per[3].item())
                    x = QZx2 - QZx1
                    y = QZy2 - QZy1
                    reQZx1 = QZx1 - x / 2
                    reQZy1 = QZy1 - y / 2
                    reQZx2 = QZx2 + x / 2
                    reQZy2 = QZy2 + y / 2
                    if (((reQZx1 < perx1 < reQZx2) and (reQZy1 < pery1 < reQZy2)) or \
                            ((reQZx1 < perx2 < reQZx2) and (reQZy1 < pery1 < reQZy2)) or \
                            ((reQZx1 < perx1 < reQZx2) and (reQZy1 < pery2 < reQZy2)) or (
                                    (reQZx1 < perx2 < reQZx2) and (reQZy1 < pery2 < reQZy2))):
                        TF_QZpeoplecross = True
        if ((QZarmList) and (PersonList)):
            for arm in QZarmList:
                armx1 = int(arm[0].item())
                army1 = int(arm[1].item())
                armx2 = int(arm[2].item())
                army2 = int(arm[3].item())
                areaarm = (armx2 - armx1) * (army2 - army1)
                for per in PersonList:
                    perx1 = int(per[0].item())
                    pery1 = int(per[1].item())
                    perx2 = int(per[2].item())
                    pery2 = int(per[3].item())
                    areaper = (perx2 - perx1) * (pery2 - pery1)
                    if (((perx1 > armx1) and (perx2 < armx2) and (pery1 > army2) and (pery2 < army2))):
                        TF_perInarm = True
                    if ((perx1 > armx1) and (perx2 < army2) and (2 * pery1 - pery2 < army2) and (
                            3.5 * areaper < areaarm) and TF_perInarm == False and ((pery1 - 30 > army2))):
                        TF_ArmPerson = True
                # 14。动土作业人员未离开
        if ((PTList) and (PersonList)):
            for PT in PTList:
                PTx1 = int(PT[0].item())
                PTy1 = int(PT[1].item())
                PTx2 = int(PT[2].item())
                PTy2 = int(PT[3].item())
                for per in PersonList:
                    perx1 = int(per[0].item())
                    pery1 = int(per[1].item())
                    perx2 = int(per[2].item())
                    pery2 = int(per[3].item())

                    x = PTx2 - PTx1
                    y = PTy2 - PTy1

                    rePTx1 = PTx1 - x / 2
                    rePTy1 = PTy1 - y / 2
                    rePTx2 = PTx2 + x / 2
                    rePTy2 = PTy2 + y / 2
                    if (((rePTx1 < perx1 < rePTx2) and (rePTy1 < pery1 < rePTy2)) or \
                            ((rePTx1 < perx2 < rePTx2) and (rePTy1 < pery1 < rePTy2)) or \
                            ((rePTx1 < perx1 < rePTx2) and (rePTy1 < pery2 < rePTy2)) or (
                                    (rePTx1 < perx2 < rePTx2) and (rePTy1 < pery2 < rePTy2))):
                        TF_PTpeoplecross = True
        # 判断用火作业违规
        if YQPList and O2PList and PersonList:
            TF_YHopration = True
            # print(TF_YHopration)
            # 11.判断是否有消防器材
            if not XFList:
                TF_YHNOxfq = True
        if YHList:
            for YH in YHList:
                YHx1 = int(YH[0].item())
                YHy1 = int(YH[1].item())
                YHx2 = int(YH[2].item())
                YHy2 = int(YH[3].item())

                # 根据HSV判断罐子类型：乙炔瓶、氧气瓶、消防器材
                imageYH = im0[YHy1:YHy2, YHx1: YHx2]  # imageYH 为罐子截图
                (b, g, r) = cv2.split(imageYH)  # 通道分解
                bH = cv2.equalizeHist(b)
                gH = cv2.equalizeHist(g)
                rH = cv2.equalizeHist(r)
                result = cv2.merge((bH, gH, rH), )  # 通道合成
                img1 = cv2.resize(result, (300, 300))
                HSV = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
                # red
                lower_r1 = np.array([155, 43, 25])
                upper_r1 = np.array([180, 255, 255])
                lower_r2 = np.array([0, 43, 25])
                upper_r2 = np.array([15, 255, 255])
                # white
                lower_w = np.array([0, 0, 221])
                upper_w = np.array([180, 30, 255])
                # blue
                lower_b = np.array([90, 30, 46])
                upper_b = np.array([120, 255, 255])
                mask_r = (cv2.inRange(HSV, lower_r1, upper_r1) + cv2.inRange(HSV, lower_r2, upper_r2))
                mask_w = (cv2.inRange(HSV, lower_w, upper_w))
                mask_b = (cv2.inRange(HSV, lower_b, upper_b))
                median_r = cv2.medianBlur(mask_r, 5)
                median_w = cv2.medianBlur(mask_w, 5)
                median_b = cv2.medianBlur(mask_b, 5)
                w = r = b = 0

                def ostu(img):
                    area = 0
                    height, width = img.shape
                    for i in range(50, 250):
                        for j in range(50, 250):
                            if img[i, j] == 255:
                                area += 1
                    value = area / (200 * 200)
                    return value

                tr = ostu(median_r)
                tw = ostu(median_w)
                tb = ostu(median_b)
                x = max(tr, tw, tb)

                if tw == x:
                    w = 1
                elif tb == x:
                    b = 1
                else:
                    r = 1
                # 如果为消防器材
                if r:
                    XFList.append(YH)
                # 如果为乙炔瓶
                if w:
                    YQPList.append(YH)
                # 如果为氧气瓶
                if b:
                    O2PList.append(YH)

        # 8 判断作业现场是否有监护人
        if ((QZList or PTList or QZarmList or YHList) and (GuardianList == []) and PersonList != []):
            TF_NOG = True

        for each in draw_dic:
            for each_cord in draw_dic[each]:
                new_img = plot_one_box(each_cord, new_img, label=each, color=[0, 0, 255], line_thickness=3)
        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))

        new_img = im0.copy()
        pp = pp + 1
        idx = -20
        weiguilist = [False * 6]
        if (TF_phone == True):
            weiguilist[0] = True
        if (TF_smoke == True):
            weiguilist[1] = True
        if (TF_NoSafeHat == True):
            weiguilist[2] = True
        if (TF_noclothes == True):
            weiguilist[3] = True
        if (TF_NOG == True):
            weiguilist[4] = True
        if (TF_wuran == True):
            weiguilist[5] = True
        if TF_phone:
            idx = idx + 30
            im0 = cv2ImgAddText(im0, "违禁使用手机", 20, idx, (255, 0, 0), 30)
            new_img = cv2ImgAddText(new_img, "违禁使用手机", 20, idx, (255, 0, 0), 30)

        if TF_smoke:
            idx = idx + 30
            im0 = cv2ImgAddText(im0, "生产区域吸烟", 20, idx, (255, 0, 0), 30)
            new_img = cv2ImgAddText(new_img, "生产区域吸烟", 20, idx, (255, 0, 0), 30)

        if TF_NoSafeHat:
            idx = idx + 30
            im0 = cv2ImgAddText(im0, "未戴安全帽", 20, idx, (255, 0, 0), 30)
            new_img = cv2ImgAddText(new_img, "未戴安全帽", 20, idx, (255, 0, 0), 30)

        if TF_noclothes:
            idx = idx + 30
            im0 = cv2ImgAddText(im0, "未穿工衣", 20, idx, (255, 0, 0), 30)
            new_img = cv2ImgAddText(new_img, "未穿工衣", 20, idx, (255, 0, 0), 30)

        if TF_QZ:
            idx = idx + 30
            im0 = cv2ImgAddText(im0, "起重作业", 20, idx, (255, 0, 0), 30)

            new_img = cv2ImgAddText(new_img, "起重作业", 20, idx, (255, 0, 0), 30)

        if TF_PT:
            idx = idx + 30
            im0 = cv2ImgAddText(im0, "动土作业", 20, idx, (255, 0, 0), 30)
            new_img = cv2ImgAddText(new_img, "动土作业", 20, idx, (255, 0, 0), 30)

        if TF_YHopration:
            idx = idx + 30
            im0 = cv2ImgAddText(im0, "用火作业", 20, idx, (255, 0, 0), 30)
            new_img = cv2ImgAddText(new_img, "用火作业", 20, idx, (255, 0, 0), 30)

        if TF_NOG:
            idx = idx + 30
            im0 = cv2ImgAddText(im0, "起重/动土/用火作业现场无监护人", 20, idx, (255, 0, 0), 30)
            new_img = cv2ImgAddText(new_img, "起重/动土/用火作业现场无监护人", 20, idx, (255, 0, 0), 30)

        if TF_wuran:
            idx = idx + 30
            im0 = cv2ImgAddText(im0, "溢油", 20, idx, (255, 0, 0), 30)
            new_img = cv2ImgAddText(new_img, "溢油", 20, idx, (255, 0, 0), 30)
        # new_img = cv2ImgAddText(new_img, "溢45245.34油", 20, idx, (255, 0, 0), 30)

        savePath = './result1/yuantu/_{}.jpg'.format(now)
        cv2.imwrite(savePath, yuantu)
        ###########################################################################
        NOW = datetime.datetime.now()
        Area = "----"

        if TF_phone:
            TF_weigui = True
            weigui = "phone"
            url = './result/phone/phone_{}.jpg'.format(now)
            dic[weigui] = url
            cate_ce[weigui] = "违禁使用手机"
            if TF_save[i] == False:
                cv2.imwrite(url, new_img)
                savePath = './result1/phone/_{}.jpg'.format(now)
                cv2.imwrite(savePath, yuantu)
                discern_post_img(cate_ce[weigui], NOW, url, Area)
        if TF_smoke:
            TF_weigui = True
            weigui = "smoke"
            url = './result/smoke/smoke_{}.jpg'.format(now)
            dic[weigui] = url
            cate_ce[weigui] = "生产区域吸烟"
            if TF_save[i] == False:
                cv2.imwrite(url, new_img)
                savePath = './result1/smoke/_{}.jpg'.format(now)
                cv2.imwrite(savePath, yuantu)
                discern_post_img(cate_ce[weigui], NOW, url, Area)

        if TF_NoSafeHat:
            TF_weigui = True
            weigui = "nosafehat"
            url = './result/nosafehat/nosafehat_{}.jpg'.format(now)
            dic[weigui] = url
            cate_ce[weigui] = "未戴安全帽"
            if TF_save[i] == False:
                cv2.imwrite(url, new_img)
                savePath = './result1/nosafehat/_{}.jpg'.format(now)
                cv2.imwrite(savePath, yuantu)
                discern_post_img(cate_ce[weigui], NOW, url, Area)

        if TF_noclothes:
            TF_weigui = True
            weigui = "noclothes"
            url = './result/noclothes/noclothes_{}.jpg'.format(now)
            dic[weigui] = url
            cate_ce[weigui] = "未穿工衣"
            if TF_save[i] == False:
                cv2.imwrite(url, new_img)
                savePath = './result1/noclothes/_{}.jpg'.format(now)
                cv2.imwrite(savePath, yuantu)
                discern_post_img(cate_ce[weigui], NOW, url, Area)

        # if TF_QZ:
        #     TF_weigui = True
        #     weigui = "QZ"
        #     url = './result/QZ/QZ_{}.jpg'.format(now)
        #     dic[weigui] = url
        #     cate_ce[weigui] = "起重作业"
        #     if TF_save[i] == False:
        #         cv2.imwrite(url, new_img)
        #         savePath = './result1/QZ/qz_{}.jpg'.format(now)
        #         cv2.imwrite(savePath, yuantu)
        #         discern_post_img(cate_ce[weigui], NOW, url, Area)
        #
        # if TF_YHopration:
        #     TF_weigui = True
        #     weigui = "YHopration"
        #     url = './result/YH/YHopration_{}.jpg'.format(now)
        #     dic[weigui] = url
        #     cate_ce[weigui] = "用火作业"
        #     if TF_save[i] == False:
        #         cv2.imwrite(url, new_img)
        #         savePath = './result1/YH/zy_{}.jpg'.format(now)
        #         cv2.imwrite(savePath, yuantu)
        #         discern_post_img(cate_ce[weigui], NOW, url, Area)
        if TF_NOG:
            TF_weigui = True
            weigui = "NOG"
            url = './result/NOG/NOG_{}.jpg'.format(now)
            dic[weigui] = url
            cate_ce[weigui] = "无监护人"
            if TF_save[i] == False:
                cv2.imwrite(url, new_img)
                savePath = './result1/NOG/_{}.jpg'.format(now)
                cv2.imwrite(savePath, yuantu)
                discern_post_img(cate_ce[weigui], NOW, url, Area)

        if TF_wuran:
            TF_weigui = True
            weigui = "wuran"
            url = './result/wuran/oil_{}.jpg'.format(now)
            dic[weigui] = url
            cate_ce[weigui] = "疑似污染"
            if TF_save[i] == False:
                cv2.imwrite(url, new_img)
                savePath = './result1/wuran/_{}.jpg'.format(now)
                cv2.imwrite(savePath, yuantu)
                discern_post_img(cate_ce[weigui], NOW, url, Area)

        isShow = True
        # if isShow:
        #     cv2.namedWindow("a", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        #     # print('!!!!!!!!!!!!!!!!!!!!!!!')
        #     cv2.imshow('a', new_img)
        #     cv2.waitKey(0)  # 1 millisecond
        # print(new_img.shape)
    # break

    return new_img, TF_weigui, weiguilist


# 一次检测
def detect_one1(path, model1, model2, area):
    half = True
    model1.half()  # to FP16
    model2.half()
    # img = cv2.imread(path)
    img = getImgByRtsp(rtsp=path)
    if img is None:
        return None, None, None
    img_size = 640
    im0 = img
    stride = int(model1.stride.max())
    img_size = check_img_size(img_size, s=stride)
    img = letterbox(im0, auto=True, new_shape=img_size, stride=32)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    # model.warmup(imgsz=(1, 3, *img_size), half=half)  # warmup
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        # img = img[None]  # expand for batch dim
        img = img.unsqueeze(0)
    pred = model1(img, augment=True)[0]
    pred = non_max_suppression(pred)
    for i in range(10):
        TF_save[i] = False
    for each in os.listdir('./ztry'):
        os.remove('./ztry/' + each)
    view_img = True
    cudnn.benchmark = True  # set True to speed up constant image size inference

    # Get names and colors
    # 获取类别名字
    names = model1.module.names if hasattr(model1, 'module') else model1.names
    print(names)
    # 设置画框的颜色
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    cate_ce = {}
    dic = {}
    pp = 0
    global person_imgg, flg, second_img

    for i, det in enumerate(pred):  # detections per image
        PhoneList.clear()
        SmokeList.clear()
        PersonList.clear()
        QZList.clear()
        PTList.clear()
        ClothList.clear()
        NoSafeHatList.clear()
        GuardianList.clear()
        OilList.clear()
        LineList.clear()
        PTpersoncrossList.clear()
        QZpersoncrossList.clear()
        CaiYouShuList.clear()
        draw_dic.clear()
        YHList = []
        KetoujiheadList = []
        KetoujiList = []
        YQPList = []
        O2PList = []
        XFList = []
        YHtljList = []
        QZarmList.clear()

        TF_noclothes = False
        TF_QZ = False
        TF_YHopration = False
        TF_PT = False
        TF_NOG = False
        TF_phone = False
        TF_smoke = False
        TF_NoSafeHat = False
        TF_wuran = False
        TF_weigui = False
        TF_perInarm = False

        # 如果输入源是webcam，则batch_size不为1，取出dataset中的一张图片
        dis_time = datetime.datetime.now()

        yuantu = im0.copy()
        # 设置打印信息(图片长宽)
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            # 调整预测框的坐标：基于resize+pad的图片的坐标-->基于原size图片的坐标
            # 此时坐标格式为xyxy
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            # 保存预测结果
            for *xyxy, conf, cls in reversed(det):

                # 在原图上画框
                if True:  # Add bbox to image
                    c = int(cls)  # integer class
                    # if (c != PhoneID and c != SmokeID):
                    label = f'{names[c]} {conf:.2f}'
                    labelx = label
                    label = label.split(' ')[0]
                    yuzhi = labelx.split(' ')[1]
                    yuzhi = float(yuzhi)

                    # im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                    if (c == peopleID and yuzhi >= 0.8):
                        PersonList.append(xyxy)
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)

                    elif (c == QZ_SuiCheDiaoID and yuzhi >= 0.85):
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                        QZList.append(xyxy)
                    elif (c == QZ_QiDiaoID and yuzhi >= 0.85):
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                        QZList.append(xyxy)

                    elif (c == YH_blueID and yuzhi >= 0.6):
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                        O2PList.append(xyxy)  # 氧气瓶
                        YHList.append(xyxy)
                    elif (c == YH_whiteID and yuzhi >= 0.6):
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                        YQPList.append(xyxy)  # 乙炔瓶
                        YHList.append(xyxy)

                    elif (c == YH_redID and yuzhi >= 0.4):
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                        XFList.append(xyxy)  # 灭火器


                    elif (c == QZ_armID and yuzhi >= 0.7):
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                        QZarmList.append(xyxy)
                    elif (c == YH_tljID and yuzhi >= 0.4):
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                        YHtljList.append(xyxy)
                    elif (c == PT_WaJueJiID and yuzhi >= 0.7):
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                        PTList.append(xyxy)

                    elif (c == ketouji_headID and yuzhi >= 0.65):
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                        KetoujiheadList.append(xyxy)
                    elif (c == ketoujiID and yuzhi >= 0.65):
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                        KetoujiList.append(xyxy)

                    elif (c == oilID and yuzhi >= 0.7):
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                        OilList.append(xyxy)

                    elif (c == flagID and yuzhi >= 0.25):
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                        LineList.append(xyxy)

        new_img = im0.copy()
        # 二次检测
        # 6 动土作业
        if PersonList and PTList:
            TF_PT = True
            # 15溢油
        if OilList:
            for oil in OilList:
                oilx1 = int(oil[0].item())  # x1
                oily1 = int(oil[1].item())  # y1
                oilx2 = int(oil[2].item())  # x2
                oily2 = int(oil[3].item())  # y2
                oilxmid = (oilx1 + oilx2) // 2
                oilymid = (oily1 + oily2) // 2
                oilimg = im0.copy()

                oil_part = []
                oil_part.append(oilimg[oily1:oilymid, oilx1:oilxmid])
                oil_part.append(oilimg[oily1:oilymid, oilxmid:oilx2])
                oil_part.append(oilimg[oilymid:oily2, oilx1:oilxmid])
                oil_part.append(oilimg[oilymid:oily2, oilxmid:oilx2])
                black_time = 0
                blue_time = 0
                for part in oil_part:
                    oil_color = get_color(part)
                    print('oil color : ', oil_color)
                    if oil_color == 'blue' or 'black':
                        black_time = black_time + 1
                        blue_time = blue_time + 1
                    # print('blacktime:---------',black_time)
                    if (black_time >= 2 or blue_time >= 3):
                        TF_wuran = True

        # 5 起重作业
        if QZList and PersonList:
            TF_QZ = True
        nosafehat2 = False
        # print(PersonList)
        if PersonList:
            num = 0

            for person in PersonList:
                personx1 = person[0]
                persony1 = person[1]
                personx2 = person[2]
                persony2 = person[3]

                x1 = int(personx1.item())
                y1 = int(persony1.item())
                x2 = int(personx2.item())
                y2 = int(persony2.item())
                person_img = im0[y1: y2, x1: x2]
                cv2.imwrite('./try/try' + str(i) + '/{}.jpg'.format(num), person_img)
                source2 = './try/try' + str(i)
                dataset2 = LoadImages(source2)
                names2 = model2.module.names if hasattr(model2, 'module') else model2.names

                # 设置画框的颜色
                colors2 = [0, 0, 255]
                for path2, img_2, im0s_2, vid_cap in dataset2:
                    if img_2 is None:
                        continue
                    img_2 = torch.from_numpy(img_2).to(device)
                    # 图片也设置为Float16
                    img_2 = img_2.half() if half else img_2.float()  # uint8 to fp16/32
                    img_2 /= 255.0  # 0 - 255 to 0.0 - 1.0
                    # 没有batch_size的话则在最前面添加一个轴
                    if img_2.ndimension() == 3:
                        img_2 = img_2.unsqueeze(0)

                    pred2 = model2(img_2, augment=True)[0]

                    pred2 = non_max_suppression(pred2)

                    im0_2 = im0s_2.copy()
                    for i2, det2 in enumerate(pred2):  # detections per image
                        # 如果输入源是webcam，则batch_size不为1，取出dataset中的一张图片

                        p2, s2, im0_2, frame = path2, '', im0s_2.copy(), getattr(dataset2, 'frame', 0)
                        if len(det2):
                            det2[:, :4] = scale_coords(img_2.shape[2:], det2[:, :4], im0_2.shape).round()
                            for *xyxy2, conf, cls2 in reversed(det2):
                                if True:  # Add bbox to image
                                    c2 = int(cls2)  # integer clas

                                    label = f'{names2[c2]} {conf:.2f}'
                                    labelx = label
                                    label = label.split(' ')[0]
                                    yuzhi = labelx.split(' ')[1]
                                    yuzhi = float(yuzhi)
                                    if (c2 == GuardianID):
                                        im0_2 = plot_one_box(xyxy2, im0_2, label=label, color=colors2, line_thickness=3)
                                        GuardianList.append(xyxy2)

                                    if c2 == No_SafeHatID and yuzhi >= 0.6:
                                        TF_NoSafeHat = True
                                        im0_2 = plot_one_box(xyxy2, im0_2, label=label, color=colors2, line_thickness=3)

                                    if (c2 == PhoneID and yuzhi >= 0.1):  # 3. 违禁使用手机
                                        im0_2 = plot_one_box(xyxy2, im0_2, label=label, color=colors2,
                                                             line_thickness=3)

                                        PhoneList.append(xyxy2)
                                        TF_phone = True
                                    if (c2 == SmokeID and yuzhi >= 0.1):  # 2 生产区域吸烟
                                        im0_2 = plot_one_box(xyxy2, im0_2, label=label, color=colors2,
                                                             line_thickness=3)
                                        SmokeList.append(xyxy2)
                                        TF_smoke = True

                    im0[y1:y2, x1:x2] = im0_2
                    new_img[y1:y2, x1:x2] = im0_2

            # 2.工衣判断
            logger.debug("工衣判断")

            now_time = datetime.datetime.now().hour
            if time.tzname[0] == 'UTC':
                now_time = (now_time + 8) % 24
            if 5 <= now_time <= 19:
                for person in PersonList:
                    imagea = im0.copy()
                    px1 = person[0]
                    py1 = person[1]
                    px2 = person[2]
                    py2 = person[3]
                    personxmin = int(px1.item()) + 5  # 获取人的图片和坐标
                    personymin = int(py1.item()) + 5
                    personxmax = int(px2.item()) - 5
                    personymax = int(py2.item()) - 5
                    if (personxmax - personxmin) < 10 or (personymax - personymin) < 20:
                        continue
                    personmidx1 = personxmin + (personxmax - personxmin) // 3
                    personmidx2 = personxmax - (personxmax - personxmin) // 3
                    personmidy = (personymin + personymax) // 2
                    person_part = []
                    # person_part.append(imagea[personmidy - personmidy//2:personmidy + personmidy//2,personmidx - personmidx//2:personmidx + personmidx//2])
                    person_part.append(imagea[personymin:personmidy, personxmin:personmidx1])
                    person_part.append(imagea[personymin:personmidy, personmidx1:personmidx2])
                    person_part.append(imagea[personymin:personmidy, personmidx2:personxmax])

                    ff = 0
                    for each_part in person_part:

                        clothes_color = get_color(each_part)
                        logger.info(clothes_color)
                        # print('********color**********', clothes_color)
                        if clothes_color == 'red' or clothes_color == 'red2':
                            ff = ff + 1
                            break
                    person_part.clear()
                    person_part.append(imagea[personmidy:personymax, personxmin:personmidx1])
                    person_part.append(imagea[personmidy:personymax, personmidx1:personmidx2])
                    person_part.append(imagea[personmidy:personymax, personmidx2:personxmax])
                    for each_part in person_part:
                        clothes_color = get_color(each_part)
                        logger.debug(clothes_color)

                        if clothes_color == 'red' or clothes_color == 'red2':
                            ff = ff + 1
                            break
                    person_part.clear()
                    if ff != 2:
                        # draw_dic.setdefault("person",[]).append(person)
                        TF_noclothes = True
                    if ff == 1 and (personxmax - personxmin) / (personymax - personymin) > 0.45:
                        TF_noclothes = False
                        # retinex----------------------------------------------------------------------
                    if TF_noclothes == True:
                        # imagea = cv2.resize(imagea, (50, 100))
                        person_img = im0[personymin:personymax, personxmin:personxmax]
                        person_img = automatedMSRCR(person_img, [15, 80, 200])
                        person_img_width = person_img.shape[1]
                        person_img_height = person_img.shape[0]
                        person_part.clear()
                        personmidx1 = person_img_width // 3
                        personmidx2 = (person_img_width // 3) * 2
                        personmidy = person_img_height // 2
                        person_part.append(person_img[0:personmidy, 0:personmidx1])
                        person_part.append(person_img[0:personmidy, personmidx1:personmidx2])
                        person_part.append(person_img[0:personmidy, personmidx2:person_img_width])
                        ff = 0
                        for each_part in person_part:
                            clothes_color = get_color(each_part)
                            logger.warning(clothes_color)

                            if clothes_color == 'red' or clothes_color == 'red2':
                                ff = ff + 1
                                break
                        person_part.clear()
                        person_part.append(person_img[personmidy:person_img_height, 0:personmidx1])
                        person_part.append(
                            person_img[personmidy:person_img_height, personmidx1:personmidx2])
                        person_part.append(
                            person_img[personmidy:person_img_height, personmidx2:person_img_width])

                        for each_part in person_part:
                            clothes_color = get_color(each_part)
                            logger.error(clothes_color)
                            if clothes_color == 'red' or clothes_color == 'red2':
                                ff = ff + 1
                                break
                        if ff == 2:
                            TF_noclothes = False
                        if ff != 2:
                            draw_dic.setdefault("person", []).append(person)
                            TF_noclothes = True
                            break
                        if ff == 1 and (personxmax - personxmin) / (personymax - personymin) > 0.45:
                            TF_noclothes = False
                    logger.error(len(PersonList))
                    logger.error("未穿工衣：" + str(TF_noclothes) + '   ' + str(ff))
                    if TF_noclothes == True:
                        break

        # 13 起重作业人员未离开
        if ((QZarmList) and (PersonList)):
            for QZ in QZarmList:
                QZx1 = int(QZ[0].item())
                QZy1 = int(QZ[1].item())
                QZx2 = int(QZ[2].item())
                QZy2 = int(QZ[3].item())
                for per in PersonList:
                    perx1 = int(per[0].item())
                    pery1 = int(per[1].item())
                    perx2 = int(per[2].item())
                    pery2 = int(per[3].item())
                    x = QZx2 - QZx1
                    y = QZy2 - QZy1
                    reQZx1 = QZx1 - x / 2
                    reQZy1 = QZy1 - y / 2
                    reQZx2 = QZx2 + x / 2
                    reQZy2 = QZy2 + y / 2
                    if (((reQZx1 < perx1 < reQZx2) and (reQZy1 < pery1 < reQZy2)) or \
                            ((reQZx1 < perx2 < reQZx2) and (reQZy1 < pery1 < reQZy2)) or \
                            ((reQZx1 < perx1 < reQZx2) and (reQZy1 < pery2 < reQZy2)) or (
                                    (reQZx1 < perx2 < reQZx2) and (reQZy1 < pery2 < reQZy2))):
                        TF_QZpeoplecross = True
        if ((QZarmList) and (PersonList)):
            for arm in QZarmList:
                armx1 = int(arm[0].item())
                army1 = int(arm[1].item())
                armx2 = int(arm[2].item())
                army2 = int(arm[3].item())
                areaarm = (armx2 - armx1) * (army2 - army1)
                for per in PersonList:
                    perx1 = int(per[0].item())
                    pery1 = int(per[1].item())
                    perx2 = int(per[2].item())
                    pery2 = int(per[3].item())
                    areaper = (perx2 - perx1) * (pery2 - pery1)
                    if (((perx1 > armx1) and (perx2 < armx2) and (pery1 > army2) and (pery2 < army2))):
                        TF_perInarm = True
                    if ((perx1 > armx1) and (perx2 < army2) and (2 * pery1 - pery2 < army2) and (
                            3.5 * areaper < areaarm) and TF_perInarm == False and ((pery1 - 30 > army2))):
                        TF_ArmPerson = True
                # 14。动土作业人员未离开
        if ((PTList) and (PersonList)):
            for PT in PTList:
                PTx1 = int(PT[0].item())
                PTy1 = int(PT[1].item())
                PTx2 = int(PT[2].item())
                PTy2 = int(PT[3].item())
                for per in PersonList:
                    perx1 = int(per[0].item())
                    pery1 = int(per[1].item())
                    perx2 = int(per[2].item())
                    pery2 = int(per[3].item())

                    x = PTx2 - PTx1
                    y = PTy2 - PTy1

                    rePTx1 = PTx1 - x / 2
                    rePTy1 = PTy1 - y / 2
                    rePTx2 = PTx2 + x / 2
                    rePTy2 = PTy2 + y / 2
                    if (((rePTx1 < perx1 < rePTx2) and (rePTy1 < pery1 < rePTy2)) or \
                            ((rePTx1 < perx2 < rePTx2) and (rePTy1 < pery1 < rePTy2)) or \
                            ((rePTx1 < perx1 < rePTx2) and (rePTy1 < pery2 < rePTy2)) or (
                                    (rePTx1 < perx2 < rePTx2) and (rePTy1 < pery2 < rePTy2))):
                        TF_PTpeoplecross = True
        # 判断用火作业违规
        if YQPList and O2PList and PersonList:
            TF_YHopration = True
            # print(TF_YHopration)
            # 11.判断是否有消防器材
            if not XFList:
                TF_YHNOxfq = True
        if YHList:
            for YH in YHList:
                YHx1 = int(YH[0].item())
                YHy1 = int(YH[1].item())
                YHx2 = int(YH[2].item())
                YHy2 = int(YH[3].item())

                # 根据HSV判断罐子类型：乙炔瓶、氧气瓶、消防器材
                imageYH = im0[YHy1:YHy2, YHx1: YHx2]  # imageYH 为罐子截图
                (b, g, r) = cv2.split(imageYH)  # 通道分解
                bH = cv2.equalizeHist(b)
                gH = cv2.equalizeHist(g)
                rH = cv2.equalizeHist(r)
                result = cv2.merge((bH, gH, rH), )  # 通道合成
                img1 = cv2.resize(result, (300, 300))
                HSV = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
                # red
                lower_r1 = np.array([155, 43, 25])
                upper_r1 = np.array([180, 255, 255])
                lower_r2 = np.array([0, 43, 25])
                upper_r2 = np.array([15, 255, 255])
                # white
                lower_w = np.array([0, 0, 221])
                upper_w = np.array([180, 30, 255])
                # blue
                lower_b = np.array([90, 30, 46])
                upper_b = np.array([120, 255, 255])
                mask_r = (cv2.inRange(HSV, lower_r1, upper_r1) + cv2.inRange(HSV, lower_r2, upper_r2))
                mask_w = (cv2.inRange(HSV, lower_w, upper_w))
                mask_b = (cv2.inRange(HSV, lower_b, upper_b))
                median_r = cv2.medianBlur(mask_r, 5)
                median_w = cv2.medianBlur(mask_w, 5)
                median_b = cv2.medianBlur(mask_b, 5)
                w = r = b = 0

                def ostu(img):
                    area = 0
                    height, width = img.shape
                    for i in range(50, 250):
                        for j in range(50, 250):
                            if img[i, j] == 255:
                                area += 1
                    value = area / (200 * 200)
                    return value

                tr = ostu(median_r)
                tw = ostu(median_w)
                tb = ostu(median_b)
                x = max(tr, tw, tb)

                if tw == x:
                    w = 1
                elif tb == x:
                    b = 1
                else:
                    r = 1
                # 如果为消防器材
                # if r:
                #     XFList.append(YH)
                # 如果为乙炔瓶
                # if w:
                #     YQPList.append(YH)
                # 如果为氧气瓶
                # if b:
                #     O2PList.append(YH)

        # 8 判断作业现场是否有监护人
        if ((QZList or PTList or QZarmList or YHList) and (GuardianList == []) and PersonList != []):
            TF_NOG = True

        for each in draw_dic:
            for each_cord in draw_dic[each]:
                new_img = plot_one_box(each_cord, new_img, label=each, color=[0, 0, 255], line_thickness=3)
        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))

        new_img = im0.copy()
        pp = pp + 1
        idx = -20
        weiguilist = {}
        if (TF_phone == True):
            weiguilist["101"] = "违禁使用手机"
        if (TF_smoke == True):
            weiguilist["102"] = "生产区域吸烟"
        if (TF_NoSafeHat == True):
            weiguilist["103"] = "未戴安全帽"
        if (TF_noclothes == True):
            weiguilist["104"] = "未穿工衣"
        if (TF_NOG == True):
            weiguilist["105"] = "作业现场无监护人"
        if (TF_wuran == True):
            weiguilist["106"] = "液体污染"
        if TF_phone:
            idx = idx + 30
            im0 = cv2ImgAddText(im0, "违禁使用手机", 20, idx, (255, 0, 0), 30)
            new_img = cv2ImgAddText(new_img, "违禁使用手机", 20, idx, (255, 0, 0), 30)

        if TF_smoke:
            idx = idx + 30
            im0 = cv2ImgAddText(im0, "生产区域吸烟", 20, idx, (255, 0, 0), 30)
            new_img = cv2ImgAddText(new_img, "生产区域吸烟", 20, idx, (255, 0, 0), 30)

        if TF_NoSafeHat:
            idx = idx + 30
            im0 = cv2ImgAddText(im0, "未戴安全帽", 20, idx, (255, 0, 0), 30)
            new_img = cv2ImgAddText(new_img, "未戴安全帽", 20, idx, (255, 0, 0), 30)

        if TF_noclothes:
            if len(GuardianList) > 0:
                pass
            else:
                idx = idx + 30
                im0 = cv2ImgAddText(im0, "未穿工衣", 20, idx, (255, 0, 0), 30)
                new_img = cv2ImgAddText(new_img, "未穿工衣", 20, idx, (255, 0, 0), 30)

        if TF_QZ:
            idx = idx + 30
            im0 = cv2ImgAddText(im0, "起重作业", 20, idx, (255, 0, 0), 30)

            new_img = cv2ImgAddText(new_img, "起重作业", 20, idx, (255, 0, 0), 30)

        if TF_PT:
            idx = idx + 30
            im0 = cv2ImgAddText(im0, "动土作业", 20, idx, (255, 0, 0), 30)
            new_img = cv2ImgAddText(new_img, "动土作业", 20, idx, (255, 0, 0), 30)

        if TF_YHopration:
            idx = idx + 30
            im0 = cv2ImgAddText(im0, "用火作业", 20, idx, (255, 0, 0), 30)
            new_img = cv2ImgAddText(new_img, "用火作业", 20, idx, (255, 0, 0), 30)

        if TF_NOG:
            idx = idx + 30
            im0 = cv2ImgAddText(im0, "起重/动土/用火作业现场无监护人", 20, idx, (255, 0, 0), 30)
            new_img = cv2ImgAddText(new_img, "起重/动土/用火作业现场无监护人", 20, idx, (255, 0, 0), 30)

        if TF_wuran:
            idx = idx + 30
            im0 = cv2ImgAddText(im0, "溢油", 20, idx, (255, 0, 0), 30)
            new_img = cv2ImgAddText(new_img, "溢油", 20, idx, (255, 0, 0), 30)
        # new_img = cv2ImgAddText(new_img, "溢45245.34油", 20, idx, (255, 0, 0), 30)

        savePath = './result1/yuantu/_{}.jpg'.format(now)
        cv2.imwrite(savePath, yuantu)
        ###########################################################################
        NOW = datetime.datetime.now()
        Area = area

        if TF_phone:
            TF_weigui = True
            weigui = "phone"
            url = './result/phone/phone_{}.jpg'.format(now)
            dic[weigui] = url
            cate_ce[weigui] = "违禁使用手机"
            if TF_save[i] == False:
                cv2.imwrite(url, new_img)
                savePath = './result1/phone/_{}.jpg'.format(now)
                cv2.imwrite(savePath, yuantu)
                discern_post_img(cate_ce[weigui], NOW, url, Area)
        if TF_smoke:
            TF_weigui = True
            weigui = "smoke"
            url = './result/smoke/smoke_{}.jpg'.format(now)
            dic[weigui] = url
            cate_ce[weigui] = "生产区域吸烟"
            if TF_save[i] == False:
                cv2.imwrite(url, new_img)
                savePath = './result1/smoke/_{}.jpg'.format(now)
                cv2.imwrite(savePath, yuantu)
                discern_post_img(cate_ce[weigui], NOW, url, Area)

        if TF_NoSafeHat:
            TF_weigui = True
            weigui = "nosafehat"
            url = './result/nosafehat/nosafehat_{}.jpg'.format(now)
            dic[weigui] = url
            cate_ce[weigui] = "未戴安全帽"
            if TF_save[i] == False:
                cv2.imwrite(url, new_img)
                savePath = './result1/nosafehat/_{}.jpg'.format(now)
                cv2.imwrite(savePath, yuantu)
                discern_post_img(cate_ce[weigui], NOW, url, Area)

        if TF_noclothes:
            if len(GuardianList) > 0:
                pass
            else:
                TF_weigui = True
                weigui = "noclothes"
                url = './result/noclothes/noclothes_{}.jpg'.format(now)
                dic[weigui] = url
                cate_ce[weigui] = "未穿工衣"
                if TF_save[i] == False:
                    cv2.imwrite(url, new_img)
                    savePath = './result1/noclothes/_{}.jpg'.format(now)
                    cv2.imwrite(savePath, yuantu)
                    discern_post_img(cate_ce[weigui], NOW, url, Area)

        # if TF_QZ:
        #     TF_weigui = True
        #     weigui = "QZ"
        #     url = './result/QZ/QZ_{}.jpg'.format(now)
        #     dic[weigui] = url
        #     cate_ce[weigui] = "起重作业"
        #     if TF_save[i] == False:
        #         cv2.imwrite(url, new_img)
        #         savePath = './result1/QZ/qz_{}.jpg'.format(now)
        #         cv2.imwrite(savePath, yuantu)
        #         discern_post_img(cate_ce[weigui], NOW, url, Area)
        #
        # if TF_YHopration:
        #     TF_weigui = True
        #     weigui = "YHopration"
        #     url = './result/YH/YHopration_{}.jpg'.format(now)
        #     dic[weigui] = url
        #     cate_ce[weigui] = "用火作业"
        #     if TF_save[i] == False:
        #         cv2.imwrite(url, new_img)
        #         savePath = './result1/YH/zy_{}.jpg'.format(now)
        #         cv2.imwrite(savePath, yuantu)
        #         discern_post_img(cate_ce[weigui], NOW, url, Area)
        if TF_NOG:
            TF_weigui = True
            weigui = "NOG"
            url = './result/NOG/NOG_{}.jpg'.format(now)
            dic[weigui] = url
            cate_ce[weigui] = "无监护人"
            if TF_save[i] == False:
                cv2.imwrite(url, new_img)
                savePath = './result1/NOG/_{}.jpg'.format(now)
                cv2.imwrite(savePath, yuantu)
                discern_post_img(cate_ce[weigui], NOW, url, Area)

        if TF_wuran:
            TF_weigui = True
            weigui = "wuran"
            url = './result/wuran/oil_{}.jpg'.format(now)
            dic[weigui] = url
            cate_ce[weigui] = "疑似污染"
            if TF_save[i] == False:
                cv2.imwrite(url, new_img)
                savePath = './result1/wuran/_{}.jpg'.format(now)
                cv2.imwrite(savePath, yuantu)
                discern_post_img(cate_ce[weigui], NOW, url, Area)

        isShow = True
        # if isShow:
        #     cv2.namedWindow("a", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        #     # print('!!!!!!!!!!!!!!!!!!!!!!!')
        #     cv2.imshow('a', new_img)
        #     cv2.waitKey(0)  # 1 millisecond
        # print(new_img.shape)
    # break
    return new_img, TF_weigui, weiguilist


a = 1


@app.route('/api/getScene', methods=['POST'])
def getScene():
    info = request.get_json()
    obj1 = threading.Thread(target=mythread, args=(info,))
    obj1.start()
    res = {"success": True}
    return json.dumps(res, ensure_ascii=False, indent=4)


# if __name__ == '__main__':
#
#
#
#     # while True:
#     with torch.no_grad():
#         save_num = detect_one1("./jz.jpg")

if __name__ == '__main__':
    logger.remove(handler_id=None)
    logger.add('api.log', rotation="200 MB")
    logger.configure()
    imgid = 0
    weights = "./newbest.pt"
    weights2 = "./1115erci.pt"
    device = torch.device('cuda', 0)
    loadmodel1 = loadmodel(weights, device)
    model1 = loadmodel1.getModel()
    loadmodel2 = loadmodel(weights2, device)
    model2 = loadmodel2.getModel()  # load FP32 model
    app.run(host='0.0.0.0', port=5590)
