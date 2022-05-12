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


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


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


class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


# 一次检测
def detect_one(save_num):
    draw_flg = True
    for i in range(10):
        TF_save[i] = False
    for each in os.listdir('./ztry'):
        os.remove('./ztry/' + each)

    source2 = opt2.source
    weights2, imgsz2 = opt2.weights, opt2.img_size
    # 获取输入、权重等参数
    source, weights, view_img, save_txt, imgsz = opt1.source, opt1.weights, opt1.view_img, opt1.save_txt, opt1.img_size
    save_img = not opt1.nosave and not source.endswith('.txt')  # save inference images
    # source="data/images"

    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # frame_provider = ImageReader(args_pose.images)
    # net = jit.load('models/openpose1.jit', map_location=device)

    # *************************************************************************
    # action_net = jit.load('models/action1.jit', map_location=device)
    # Directories
    save_dir = increment_path(Path(opt1.project) / opt1.name, exist_ok=opt1.exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize

    device = select_device(opt1.device)
    # 如果设备为gpu，使用Float16
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model2 = attempt_load(weights2, map_location=device)  # load FP32 model
    stride2 = int(model2.stride.max())  # model stride
    imgsz2 = check_img_size(imgsz2, s=stride2)  # check img_size
    # Load model
    # 加载float模型，确定用户设置的输入图片分辨率能整除32(如不能则调整为能整除并返回)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
    # 设置Float16
    if half:
        model.half()  # to FP16
        model2.half()

    view_img = True
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams3(source, img_size=imgsz, stride=stride, channels=1)

    # Get names and colors
    # 获取类别名字
    names = model.module.names if hasattr(model, 'module') else model.names
    print(names)
    # 设置画框的颜色
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    cate_ce = {}
    dic = {}
    cntt = 0
    timeF = 1  # 磕头机停抽跳帧检测
    pp = 0
    successNum=0
    allNum=0
    global person_imgg, flg, second_img
    cc = 0
    for path, img, im0s, frame_list, area in dataset:
        allNum=allNum+1
        if img is not None:
            successNum=successNum+1
        if allNum%1==0:
            memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024
            cpu = psutil.cpu_percent()
            logger.debug(
                "内存占用：" + str(memory) + "  cpu占用：" + str(cpu) + "   " + str(psutil.Process(os.getpid()).cpu_percent()))
            logger.debug("轮询摄像头数 allNum:{},successNum:{}".format(allNum,successNum))
        if path == None:
            continue
        # cc=cc+1
        if cc % 1 == 0:
            # print("path:")
            if img is None:
                continue

            # if type(img) == bool or type(frame_list) == bool:
            #     break
            # if path == None:
            #     continue
            for each in os.listdir('./runs/detect'):
                shutil.rmtree('./runs/detect/' + each)
            img = torch.from_numpy(img).to(device)
            # 图片也设置为Float16
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            # 没有batch_size的话则在最前面添加一个轴
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred = model(img, augment=opt1.augment)[0]
            pred = non_max_suppression(pred, opt1.conf_thres, opt1.iou_thres, classes=opt1.classes,
                                       agnostic=opt1.agnostic_nms)
            # Apply Classifier
            # names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            # print(names)
            # 添加二次分类，默认不使用

            # Process detections
            # 对每一张图片作处理
            memory=psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024
            cpu=psutil.cpu_percent()
            logger.debug("内存占用："+str(memory)+"  cpu占用："+str(cpu)+"   "+str(psutil.Process(os.getpid()).cpu_percent()))
            for i, det in enumerate(pred):  # detections per image
                rtsp_adress = path[i]
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
                O2PDOWNList = []
                KEYPointsList = []
                FAIRList = []
                SmokeFogList = []
                QZarmList.clear()

                TF_noclothes = False
                TF_QZpeoplecross = False
                TF_PTpeoplecross = False
                TF_QZ = False
                TF_YHopration = False
                TF_YHNOxfq = False
                TF_YHYQPwofang = False
                TF_YHO2Pwofang = False
                TF_YHTongCheYunShu = False
                TF_JJBZ = False
                TF_YHJDQP = False
                TF_PT = False
                TF_NOG = False
                TF_phone = False
                TF_smoke = False
                TF_NoSafeHat = False
                TF_wuran = False
                TF_noline = False
                TF_NOGuardian = False
                TF_PersonCaiyoushu = False
                TF_ketoujistop = False
                TF_GHGQpeoplegather = False
                TF_FAIR = False
                TF_SMOKEFOG = False
                TF_ArmPerson = False
                TF_perInarm = False

                # 如果输入源是webcam，则batch_size不为1，取出dataset中的一张图片
                dis_time = datetime.datetime.now()
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

                yuantu = im0.copy()
                k_pic = im0.copy()
                # cv2.imshow("new",k_pic)
                p = Path(p)  # to Path
                # 设置保存图片/视频的路径
                save_path = str(save_dir / p.name)  # img.jpg
                # 设置保存框坐标txt文件的路径
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                # 设置打印信息(图片长宽)
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    # 调整预测框的坐标：基于resize+pad的图片的坐标-->基于原size图片的坐标
                    # 此时坐标格式为xyxy
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    # 打印检测到的类别数量
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    # 保存预测结果
                    det_names=[]
                    for *xyxy, conf, cls in reversed(det):
                        det_names.append(names[int(cls)])
                        det_names.append(conf)
                        if save_txt:  # Write to file
                            # 将xyxy(左上角+右下角)格式转为xywh(中心点+宽长)格式，并除上w，h做归一化，转化为列表再保存
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt1.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        # 在原图上画框
                        if save_img or opt1.save_crop or view_img:  # Add bbox to image
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
                            #elif ((c == YH_blueID or c == YH_whiteID) and yuzhi >= 0.4):
                                #im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                                #YHList.append(xyxy)
                            elif (c == YH_blueID and yuzhi >= 0.6):
                                im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                                O2PList.append(xyxy) #氧气瓶
                                YHList.append(xyxy)
                            elif ( c == YH_whiteID and yuzhi >= 0.6):
                                im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                                YQPList.append(xyxy) #乙炔瓶
                                YHList.append(xyxy)

                            elif (c == YH_redID and yuzhi >= 0.4):
                                im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                                XFList.append(xyxy) #灭火器


                            elif(c== QZ_armID and yuzhi >= 0.7):
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
                            # elif (c == FAIRID and yuzhi >= 1.1):
                            #     im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                            #     FAIRList.append(xyxy)
                            # elif (c == SMOKEFOGID and yuzhi >=1.1):
                            #     im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                            #     SmokeFogList.append(xyxy)

                            # elif (c == caiyoushuID and yuzhi >= 0.5):
                            # x = int(xyxy[0].item())
                            # y = int(xyxy[1].item())
                            # w = int(xyxy[2].item())
                            # h = int(xyxy[3].item())
                            # if (w * h > 500):
                            #     xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
                            #     wid = (xmax - xmin)
                            #     height = (ymax - ymin)
                            #     if ((xmin - wid) < 1) or ((xmax + wid) > im0.shape[1]) or ((ymin - height) < 1) or (
                            #             (ymax + height) > im0.shape[0]):
                            #         Caiyoushutemp = [xmin, ymin, xmax, ymax]
                            #     else:
                            #         Caiyoushutemp = [xmin - wid, ymin - height, xmax + wid, ymax + height]
                            #     CaiYouShuList.append(xyxy)
                            #     im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                            # elif(yuzhi >= 0.8):
                            #    im0 = plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                    logger.debug("一次检测结果："+str(det_names))
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
                logger.debug("人：" + str(PersonList))

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
                        dataset2 = LoadImages(source2, img_size=imgsz2, stride=stride2)
                        names2 = model2.module.names if hasattr(model2, 'module') else model2.names
                        # print(names2)
                        # if person_img is not None:
                        # falldown_state,keypoints = run_demo(net, action_net, frame_provider,
                        # args_pose.height_size, args_pose.cpu,person_img)
                        # for pum in range(18):
                        #     if keypoints[pum, 0] != -1:
                        #         keypoints[pum, 0] += x1
                        #     if keypoints[pum, 1] != -1:
                        #         keypoints[pum, 1] += y1
                        # KEYPointsList.append(keypoints)
                        # print(falldown_state)

                        # 设置画框的颜色
                        colors2 = [0, 0, 255]
                        if device.type != 'cpu':
                            model2(torch.zeros(1, 3, imgsz2, imgsz2).to(device).type_as(
                                next(model2.parameters())))  # run once
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

                            pred2 = model2(img_2, augment=opt2.augment)[0]

                            pred2 = non_max_suppression(pred2, opt2.conf_thres, opt2.iou_thres, classes=opt2.classes,
                                                        agnostic=opt2.agnostic_nms)
                            head = None
                            im0_2 = im0s_2.copy()
                            for i2, det2 in enumerate(pred2):  # detections per image
                                # 如果输入源是webcam，则batch_size不为1，取出dataset中的一张图片

                                p2, s2, im0_2, frame = path2, '', im0s_2.copy(), getattr(dataset2, 'frame', 0)
                                if len(det2):
                                    det_names2=[]
                                    det2[:, :4] = scale_coords(img_2.shape[2:], det2[:, :4], im0_2.shape).round()
                                    for *xyxy2, conf, cls2 in reversed(det2):
                                        det_names2.append(names2[int(cls2)])
                                        if save_img or opt2.save_crop or view_img:  # Add bbox to image
                                            c2 = int(cls2)  # integer clas

                                            label = f'{names2[c2]} {conf:.2f}'
                                            labelx = label
                                            label = label.split(' ')[0]
                                            yuzhi = labelx.split(' ')[1]
                                            yuzhi = float(yuzhi)
                                            if(c2 == GuardianID):
                                                im0_2 = plot_one_box(xyxy2, im0_2, label=label, color=colors2,line_thickness=3)
                                                GuardianList.append(xyxy2)

                                            if c2 == No_SafeHatID and yuzhi >= 0.6:

                                                    TF_NoSafeHat = True
                                                    im0_2 = plot_one_box(xyxy2, im0_2, label=label, color=colors2,line_thickness=3)

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
                                    logger.debug("二次检测结果"+str(det_names2))

                                            # im0_2 = plot_one_box(xyxy2, im0_2, label=label, color=colors2, line_thickness=3)

                            # 2.生产区域吸烟
                            # if SmokeList and head:
                            #     smoke_mid_y = (int(SmokeList[0][1].item()) + int(SmokeList[0][3].item())) / 2
                            #     head_y1 = int(head[1].item())
                            #     head_y2 = int(head[3].item())
                            #     if head_y1 < smoke_mid_y < head_y2:
                            #         TF_somke = True
                            #
                            #         im0_2 = plot_one_box(SmokeList[0], im0_2, label=label, color=colors2,
                            #                              line_thickness=3)
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
                            logger.error("未穿工衣："+str(TF_noclothes)+'   '+str(ff))
                            if TF_noclothes==True:
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
                        #如果为消防器材
                        if r:
                            XFList.append(YH)
                        #如果为乙炔瓶
                        if w:
                            YQPList.append(YH)
                        #如果为氧气瓶
                        if b:
                            O2PList.append(YH)


                # 判断是否有火焰烟雾
                if FAIRList:
                    TF_FAIR = True
                if SmokeFogList:
                    TF_SMOKEFOG = True

                # 8 判断作业现场是否有监护人
                if ((QZList or PTList or QZarmList or YHList) and (GuardianList == []) and PersonList != []):
                    TF_NOG = True
                # # 9 判断起重动土用火作业现场是否有警戒线
                # if (LineList == [] and ((YQPList and O2PList) or PTList or QZList) and PersonList):
                #     TF_noline = True

                    # 新加 判断人站在采油树上
                # print(CaiYouShuList,PersonList,"dddddddddddddddddddddd")
                if ((CaiYouShuList) and PersonList):
                    for cyshu in CaiYouShuList:
                        cyshux1 = int(cyshu[0].item())
                        cyshuy1 = int(cyshu[1].item())
                        cyshux2 = int(cyshu[2].item())
                        cyshuy2 = int(cyshu[3].item())
                        cyshuyy = (cyshuy2 - cyshuy1) // 3
                        cyshuyy1 = (cyshuy2 - cyshuy1) // 7

                        for per in PersonList:
                            perx1 = int(per[0].item())
                            pery1 = int(per[1].item())
                            perx2 = int(per[2].item())
                            pery2 = int(per[3].item())
                            perxx = (perx1 + perx2) // 2
                            if ((cyshux1 < perxx < cyshux2) & (cyshuy1 + cyshuyy1 < pery2 < (cyshuy2))):
                                TF_PersonCaiyoushu = True
                        # print(TF_PersonCaiyoushu)
                # print(KetoujiheadList, "LLLLLLLLLLLLLLLLLLL")

                # 高后果区人员聚集
                if len(PersonList) > 7 and OilList:
                    Oil_s = 0
                    for Oil in OilList:
                        Oilx1 = int(Oil[0].item())
                        Oily1 = int(Oil[1].item())
                        Oilx2 = int(Oil[2].item())
                        Oily2 = int(Oil[3].item())
                        Oil_area = (Oilx2 - Oilx1) * (Oily2 - Oily1)
                        # print(Oil_area)
                        Oil_s = Oil_s + Oil_area
                    # print(Oil_s)
                    if Oil_s > 9000:
                        TF_GHGQpeoplegather = True
                # print(TF_GHGQpeoplegather)

                for each in draw_dic:
                    for each_cord in draw_dic[each]:
                        new_img = plot_one_box(each_cord, new_img, label=each, color=[0, 0, 255], line_thickness=3)
                now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
                readVideoPath = opt1.source
                new_img = im0.copy()
                pp = pp + 1
                idx = -20

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
                    if len(GuardianList)>0:
                        pass
                    else:
                        idx = idx + 30
                        im0 = cv2ImgAddText(im0, "未穿工衣", 20, idx, (255, 0, 0), 30)
                        new_img = cv2ImgAddText(new_img, "未穿工衣", 20, idx, (255, 0, 0), 30)

                if TF_QZ:
                    idx = idx + 30
                    im0 = cv2ImgAddText(im0, "起重作业", 20, idx, (255, 0, 0), 30)

                    new_img = cv2ImgAddText(new_img, "起重作业", 20, idx, (255, 0, 0), 30)

                if TF_ArmPerson:
                    idx = idx + 30
                    im0 = cv2ImgAddText(im0, "吊臂下站人", 20, idx, (255, 0, 0), 30)
                    new_img = cv2ImgAddText(new_img, "吊臂下站人", 20, idx, (255, 0, 0), 30)

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

                if TF_noline:
                    idx = idx + 30
                    im0 = cv2ImgAddText(im0, "起重/动土/用火作业现场无警戒线", 20, idx, (255, 0, 0), 30)
                    new_img = cv2ImgAddText(new_img, "起重/动土/用火作业现场无警戒线", 20, idx, (255, 0, 0), 30)


                if TF_QZpeoplecross:
                    idx = idx + 30
                    im0 = cv2ImgAddText(im0, "起重作业人员未离开", 20, idx, (255, 0, 0), 30)
                    new_img = cv2ImgAddText(new_img, "起重作业人员未离开", 20, idx, (255, 0, 0), 30)

                    # if TF_PTpeoplecross:
                    idx = idx + 30
                    im0 = cv2ImgAddText(im0, "动土作业人员未离开", 20, idx, (255, 0, 0), 30)
                    new_img = cv2ImgAddText(new_img, "动土作业人员未离开", 20, idx, (255, 0, 0), 30)

                if TF_PersonCaiyoushu:
                    idx = idx + 30
                    im0 = cv2ImgAddText(im0, "人站在采油树上", 20, idx, (255, 0, 0), 30)
                    new_img = cv2ImgAddText(new_img, "人站在采油树上", 20, idx, (255, 0, 0), 30)

                if TF_ketoujistop:
                    idx = idx + 30
                    im0 = cv2ImgAddText(im0, "磕头机停抽", 20, idx, (255, 0, 0), 30)
                    new_img = cv2ImgAddText(new_img, "磕头机停抽", 20, idx, (255, 0, 0), 30)

                if TF_wuran:
                    idx = idx + 30
                    im0 = cv2ImgAddText(im0, "溢油", 20, idx, (255, 0, 0), 30)
                    new_img = cv2ImgAddText(new_img, "溢油", 20, idx, (255, 0, 0), 30)
                # new_img = cv2ImgAddText(new_img, "溢45245.34油", 20, idx, (255, 0, 0), 30)

                if TF_GHGQpeoplegather:
                    idx = idx + 30
                    im0 = cv2ImgAddText(im0, "高后果区人员聚集", 20, idx, (255, 0, 0), 30)
                    new_img = cv2ImgAddText(new_img, "高后果区人员聚集", 20, idx, (255, 0, 0), 30)

                if TF_FAIR:
                    idx = idx + 30
                    im0 = cv2ImgAddText(im0, "火焰", 20, idx, (255, 0, 0), 30)
                    new_img = cv2ImgAddText(new_img, "火焰", 20, idx, (255, 0, 0), 30)

                if TF_SMOKEFOG:
                    idx = idx + 30
                    im0 = cv2ImgAddText(im0, "烟雾", 20, idx, (255, 0, 0), 30)
                    new_img = cv2ImgAddText(new_img, "烟雾", 20, idx, (255, 0, 0), 30)
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

                if  TF_noclothes :
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


                if TF_ArmPerson:
                    TF_weigui = True
                    weigui = "diaobiPerson"
                    idx = idx + 30
                    url = './result/ArmPerson/personUnderArm_{}.jpg'.format(now)
                    dic[weigui] = url
                    cate_ce[weigui] = "吊臂下站人"
                    if TF_save[i] == False:
                        cv2.imwrite(url, new_img)
                        savePath = './result1/ArmPerson/_{}.jpg'.format(now)
                        cv2.imwrite(savePath, yuantu)
                        discern_post_img(cate_ce[weigui], NOW, url, Area)

                if TF_QZ:
                    TF_weigui = True
                    weigui = "QZ"
                    url = './result/QZ/QZ_{}.jpg'.format(now)
                    dic[weigui] = url
                    cate_ce[weigui] = "起重作业"
                    if TF_save[i] == False:
                        cv2.imwrite(url, new_img)
                        savePath = './result1/QZ/qz_{}.jpg'.format(now)
                        cv2.imwrite(savePath, yuantu)
                        discern_post_img(cate_ce[weigui], NOW, url, Area)

                if TF_YHopration:
                    TF_weigui = True
                    weigui = "YHopration"
                    url = './result/YH/YHopration_{}.jpg'.format(now)
                    dic[weigui] = url
                    cate_ce[weigui] = "用火作业"
                    if TF_save[i] == False:
                        cv2.imwrite(url, new_img)
                        savePath = './result1/YH/zy_{}.jpg'.format(now)
                        cv2.imwrite(savePath, yuantu)
                        discern_post_img(cate_ce[weigui], NOW, url, Area)

                if TF_YHNOxfq:
                    TF_weigui = True
                    weigui = "YHNOxfq"
                    url = './result/YH/YHNOxfq_{}.jpg'.format(now)
                    dic[weigui] = url
                    cate_ce[weigui] = "无消防器材"
                    if TF_save[i] == False:
                        cv2.imwrite(url, new_img)
                        savePath = './result1/YH/wuxiaofang_{}.jpg'.format(now)
                        cv2.imwrite(savePath, yuantu)
                        discern_post_img(cate_ce[weigui], NOW, url, Area)




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


                if TF_PersonCaiyoushu:
                    TF_weigui = True
                    weigui = "peple_on_caiyoushu"
                    idx = idx + 30
                    url = './result/people_on_caiyoushu/peple_on_caiyoushu{}.jpg'.format(now)
                    dic[weigui] = url
                    cate_ce[weigui] = "人站在采油树上"
                    cv2.imwrite(url, new_img)
                    savePath = './result1/people_on_caiyoushu/_{}.jpg'.format(now)
                    cv2.imwrite(savePath, yuantu)
                    discern_post_img(cate_ce[weigui], NOW, url, Area)

                if TF_ketoujistop:
                    TF_weigui = True
                    weigui = "ketouji_stop"
                    idx = idx + 30
                    url = './result/ketouji_stop/ketouji_stop{}.jpg'.format(now)
                    dic[weigui] = url
                    cate_ce[weigui] = "磕头机停抽"
                    cv2.imwrite(url, new_img)
                    savePath = './result1/ketouji_stop/_{}.jpg'.format(now)
                    cv2.imwrite(savePath, yuantu)
                    discern_post_img(cate_ce[weigui], NOW, url, Area)

                if TF_GHGQpeoplegather:
                    TF_weigui = True
                    weigui = "GHGQpeoplegather"
                    idx = idx + 30
                    url = './result/GHGQpeoplegather/GHGQpeoplegather_{}.jpg'.format(now)
                    dic[weigui] = url
                    cate_ce[weigui] = "高后果区人员聚集"
                    if TF_save[i] == False:
                        cv2.imwrite(url, new_img)
                        savePath = './result1/GHGQpeoplegather/_{}.jpg'.format(now)
                        cv2.imwrite(savePath, yuantu)
                        discern_post_img(cate_ce[weigui], NOW, url, Area)


                isShow = False
                if isShow:
                    cv2.namedWindow("a", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    # print('!!!!!!!!!!!!!!!!!!!!!!!')
                    cv2.imshow('a', new_img)
                    cv2.waitKey(500)  # 1 millisecond
                # print(new_img.shape)
            # break
    # return save_num


if __name__ == '__main__':

    # save_model = False
    # if (len(sys.argv)) == 1:
    #     save_model = False
    # else:
    #     save_model = True
    # 一次权重模型参数设置
    parser1 = argparse.ArgumentParser()
    parser1.add_argument('--weights', nargs='+', type=str, default='newbest.pt', help='model.pt path(s)')
    # parser1.add_argument('--source', type=str, default='loop.txt', help='source')
    # parser1.add_argument('--source', type=str, default='rtsp://admin:1234567a@10.66.153.239:554/Streaming/Channels/101', help='source')  # file/folder, 0 for webcam
    # parser1.add_argument('--source', type=str, default='rtsp://admin:Sfjkadmin@10.69.107.224:554/Streaming/Channels/101',help='source')  # file/folder, 0 for webcam
    # parser1.add_argument('--source', type=str, default='rtsp://10.69.105.240:554/10.69.105.240:554:8000:HIK-DS8000HC:0:0:skaiglq:Hik12345/av_stream', help='source')  # file/folder, 0 for webcam
    # parser1.add_argument('--source', type=str, default='rtsp://10.67.144.238:554/pag://192.168.145.252:7302:001485:0:MAIN:TCP', help='source')  # file/folder, 0 for webcam
    # parser1.add_argument('--source', type=str, default='rtsp://10.69.105.240:554/pag://192.100.87.67:8000:002464:0:MAIN:TCP', help='source')  # file/folder, 0 for webcam
    # parser1.add_argument('--source', type=str, default=r'C:\Users\Administrator\IdeaProjects\maven-camera\test.txt', help='source')  # file/folder, 0 for webcam
    #parser1.add_argument('--source', type=str, default=r'rtsp://10.67.206.220:554/openUrl/C7BTDb2',help='source')  # file/folder, 0 for webcam

    parser1.add_argument('--source', type=str, default=r'root000000',help = 'source')  # file/folder, 0 for webcam
    parser1.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser1.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser1.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser1.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser1.add_argument('--view-img', action='store_true', help='display results')
    parser1.add_argument('--view-img', default='false', help='display results')
    parser1.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser1.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser1.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser1.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser1.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser1.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser1.add_argument('--augment', action='store_true', help='augmented inference')
    parser1.add_argument('--update', action='store_true', help='update all models')
    parser1.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser1.add_argument('--name', default='exp', help='save results to project/name')
    parser1.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser1.add_argument('--save-data', type=int, default=0, help='save data')
    opt1 = parser1.parse_args()
    print(opt1)

    # 二次权重模型参数设置
    parser2 = argparse.ArgumentParser()
    parser2.add_argument('--weights', nargs='+', type=str, default='1115erci.pt', help='model.pt path(s)')
    parser2.add_argument('--source', type=str, default='./ztry', help='source')  # file/folder, 0 for webcam
    parser2.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser2.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser2.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser2.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser2.add_argument('--view-img', action='store_true', help='display results')
    parser2.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser2.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser2.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser2.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser2.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser2.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser2.add_argument('--augment', action='store_true', help='augmented inference')
    parser2.add_argument('--update', action='store_true', help='update all models')
    parser2.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser2.add_argument('--name', default='exp', help='save results to project/name')
    parser2.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    opt2 = parser2.parse_args()
    print(opt2)
    # parser3 = argparse.ArgumentParser(
    #     description='''Lightweight human pose estimation python demo.
    #                            This is just for quick results preview.
    #                            Please, consider c++ demo for the best performance.''')
    # parser3.add_argument('--checkpoint-path', type=str, default='checkpoint_iter_370000.pth',
    #                      required=False,
    #                      help='path weight')
    # parser3.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    # parser3.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    # parser3.add_argument('--video', type=str, default='rtsp://admin:hkv516001@192.168.0.4:554/Streaming/Channels/101',
    #                      help='path to video file or camera id')
    # parser3.add_argument('--images', nargs='+', default='', help='path to input image(s)')
    # parser3.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    # parser3.add_argument('--track', type=int, default=0, help='track pose id in video')
    # parser3.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    # opt3 = parser3.parse_args()
    #
    # if opt3.video == '' and opt3.images == '':
    #     raise ValueError('Either --video or --image has to be provided')
    # # opt3.checkpoint_path = "" + '/checkpoint_iter_370000.pth'
    # net = PoseEstimationWithMobileNet()
    # checkpoint = torch.load(opt3.checkpoint_path, map_location='cpu')
    # load_state(net, checkpoint)

    # frame_provider = ImageReader(opt3.images)
    # if opt3.video != '':
    #     frame_provider = VideoReader(opt3.video)
    # else:
    #     opt3.track = 0

    check_requirements(exclude=('pycocotools', 'thop'))
    logger.remove(handler_id=None)
    logger.add('caiyou.log', rotation="200 MB")
    logger.configure()

    # while True:
    with torch.no_grad():
        save_num = detect_one(save_num)
