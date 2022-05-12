import datetime
import logging
import threading
import time

import cv2
from loguru import logger

from utils.general import clean_str
from RTSP.getRtspByRegion import getRtsp, getCameraByRegion, getCameraAndNameByRegion
from RTSP.getRtspByRegion import getAllCamera

class LoadVideo:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=640, stride=32, channels=1, threads_num=2):
        self.success = None
        self.imgs = None
        self.sources = sources
        self.img_size = img_size
        self.stride = stride
        self.channels = channels
        self.readNum = 0
        self.camera = {}
        self.rect = True
        self.now = datetime.datetime.now()
        self.img_dict = {}
        self.threads_num = threads_num
        self.threads = [None] * threads_num
        self.camera = getCameraAndNameByRegion(sources)
    # @func_set_timeout(5)
    # @time_out(2, call_back=call_back_func, error_back=error_back_func)
    def readRtsp(self, rtsp,area):
        self.imgs = None
        cap = cv2.VideoCapture(rtsp)
        if cap is not None:
            self.readNum = 0
            self.success = False
            while True:
                try:
                    self.readNum += 1
                    ret, self.imgs = cap.read()
                    if (self.imgs is
                            not None):
                        self.img_dict[area] = self.imgs
                        break
                    if self.readNum >= 20:
                        logging.warning("not get!")
                        break
                except:
                    pass
            cap.release()

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        imgDict = {}
        for i in range(self.threads_num):
            self.count += 1
            # 轮询完重置
            if self.count == len(self.camera):
                self.count = 0
            area = list(self.camera)[self.count]
            rtsp = getRtsp(self.camera[area])
            # cap=cv2.VideoCapture(rtsp)
            # rtsp = clean_str(rtsp)
            t = threading.Thread(target=self.readRtsp, args=(rtsp,area,))
            t.start()
        time.sleep(4)

        print("运行的线程数：", len(threading.enumerate()))
        # print("//////////////////////////////////////////////////////////////////////////////////////")
        # for i in threading.enumerate():
        #     print(i.getName())
        # print("//////////////////////////////////////////////////////////////////////////////////////")
        print("@@@@@@@@@@@@@@@@@@@@:", len(self.img_dict))
        if len(threading.enumerate()) > 20:
            time.sleep(20)
        idict = list(self.img_dict.keys())
        print(idict)
        for area in idict:
            imgDict[area] = self.img_dict.pop(area)

        return imgDict
    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


if __name__ == '__main__':


    logger.error("dasdasda")
    mLoadVideo = LoadVideo("root000000")

    logger.debug("dasdasd")
    # mLoadVideo = LoadVideo(sources=["1.mp4","3.mp4","5.mp4", "2.mp4", "4.mp4"])
    num = 0
    success = 0
    for imgDict in mLoadVideo:
        print("rtsp:::", imgDict.keys())
        num = num + 1
        for i in imgDict:
            success = success + 1
        print(num, success)
        # if num == 45:
        #     time.sleep(60)
        # if num == 46:
        #     break
        # if img0 is not None:
        #     success = success + 1
        #     cv2.imshow("aa", img0)
        #     cv2.waitKey(1000)
        #     cv2.destroyAllWindows()
        # print("轮询次数{}，成功获取个数{}".format(num, success))
