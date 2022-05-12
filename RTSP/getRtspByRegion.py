from RTSP.getCameraIndex import getCameraIndex, getCameraIndexAndName
from RTSP.getSubRegions import getLastRegion
from loguru import logger
import requests
import configparser
from RTSP.OpenApi_public_def import Signature
import json
def getRtspByCamera(cameraIndex):
    # Step1：从配置文件获取host地址、端口号、appKey和appSecret
    # api config
    cf = configparser.ConfigParser()
    cf.read(".\RTSP\ApiConfig.ini")  # 读取ApiConfig.ini配置文件

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
    api = '/api/video/v1/cameras/previewURLs'  # api 的url
    methon = 'POST'  # POST 或 GET 请求

    # Step3：组装POST请求URL
    # Setting Url
    url = host + ':' + port + '/' + content + api

    # Step4：获取安全认证的Headers
    # Setting Headers
    header_dict = Signature(appSecret, methon, appKey, content, api)
    # Step5：组装传入的Json
    # Setting JSON Body
    payload = {
    "cameraIndexCode": cameraIndex,
    "streamType": 0,
    "protocol": "rtsp",
    "transmode": 1,
    "expand": "streamform=rtp"
}

    # Step6：发起POST请求
    # Make the requests
    try:
        r = requests.post(url, headers=header_dict, json=payload, verify=False)
        return r.content.decode('utf-8')
    except Exception as error:
        logger.debug(error)
    # Step7：解析请求响应
    # Check the response
        return None
#根据区域获取下级所有摄像仪编号
def getCameraByRegion(regionIndex):
    aa=getLastRegion(regionIndex)
    print(aa)
    allCameras = getCameraIndex(aa)
    return allCameras

def getCameraAndNameByRegion(regionIndex):
    allCameras=getCameraIndexAndName(getLastRegion(regionIndex))
    return allCameras
#根据摄像头编号获取RTSp
def getRtsp(cameraIndex):
    aa=getRtspByCamera(cameraIndex)
    if aa is not None:
        aa=json.loads(aa)
        return aa["data"]["url"]
    else:
        return None
#获取区域下所有摄像头编号
def getAllCamera(region):
    return getCameraIndex(getLastRegion(region))
if __name__ == '__main__':
    aa=getCameraIndex(getLastRegion("root000000"))
    print(aa)
    for i in aa:
       print(getRtsp(i))