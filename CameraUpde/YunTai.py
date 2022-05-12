import requests
import configparser
import cv2
from CameraUpde.getRtspByRegion import getRtsp
from RTSP.OpenApi_public_def import Signature
import json

'''

'''
def setCameraZoom(cameraIndex,x1,y1,x2,y2):
    # Step1：从配置文件获取host地址、端口号、appKey和appSecret
    # api config
    cf = configparser.ConfigParser()
    cf.read(".\ApiConfig.ini")  # 读取ApiConfig.ini配置文件

    host = cf.get("api-config", "host")  # 获取[api-config]中host对应的值
    # print(host)

    port = cf.get("api-config", "port")  # 获取[api-config]中port对应的值
    # print(port)

    appKey = cf.get("api-config", "appKey")  # 获取[api-config]中appKey对应的值
    # print(appKey)

    appSecret = cf.get("api-config", "appSecret")  # 获取[api-config]中appSecret对应的值
    # print(appSecret)

    # Step2：设置接口地址及请求方式
    # artemis api
    content = 'artemis'  # 上下文 默认artemis
    api = '/api/video/v1/ptzs/selZoom'  # api 的url
    methon = 'POST'  # POST 或 GET 请求

    # Step3：组装POST请求URL
    # Setting Url
    url = host + ':' + port + '/' + content + api

    # Step4：获取安全认证的Headers
    # Setting Headers
    header_dict = Signature(appSecret, methon, appKey, content, api)

    # Step5：组装传入的Json
    # Setting JSON Body
    payload ={
    "cameraIndexCode": cameraIndex,
    "startX": x1,
    "startY": y1,
    "endX": x2,
    "endY": y2
    }


    # Step6：发起POST请求
    # Make the requests
    r = requests.post(url, headers=header_dict, json=payload, verify=False)

    # Step7：解析请求响应
    # Check the response
    return r.content.decode('utf-8')

if __name__ == '__main__':
    cameraIndex="37170308411319845850"
    rtsp =getRtsp(cameraIndex)
    x1 = 120.5657
    y1 = 100.453
    x2 = 130.342
    y2 = 110.234253
    print("1")
    # cap = cv2.VideoCapture(rtsp)
    print("2")
    info=setCameraZoom(cameraIndex,x1,y1,x2,y2)
    # print("start!")
    # while True:
    #     try:
    #         ret,img=cap.read()
    #     except Exception as e:
    #         print(e)
    #     if img is None:
    #         continue
    #     cv2.imshow("aa",img)
    #
    #     cv2.waitKey(1)


    print(info)