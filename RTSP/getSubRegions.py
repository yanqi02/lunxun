# -*- coding: UTF-8 -*-
import requests
import configparser
from RTSP.OpenApi_public_def import Signature
import json
from RTSP.getCameraIndex import getCameraIndex
"""
获取下级区域列表
return:
{
    "code": "0",
    "msg": "SUCCESS",
    "data": {
        "total": 18,
        "pageNo": 1,
        "pageSize": 1,
        "list": [
            {
                "indexCode": "6e74e23d-8e4f-475d-a8b6-5f44e7161ac3",
                "name": "测试2",
                "parentIndexCode": "root000000",
                "catalogType": 10,
                "externalIndexCode": "11010508582160000029",
                "sort": 2,
                "regionPath": "@root000000@6e74e23d-8e4f-475d-a8b6-5f44e7161ac3@",
                "createTime": "2019-07-16T09:50:50.308+08:00",
                "updateTime": "2019-07-31T14:34:39.272+08:00",
                "available": true,
                "cascadeCode": "0",
                "cascadeType": 0,
                "leaf": true
            }
        ]
    }
}

"""
def getSubRegion(regionIndex):
    # Step1：从配置文件获取host地址、端口号、appKey和appSecret
    # api config
    cf = configparser.ConfigParser()
    cf.read(".\RTSP\ApiConfig.ini")  # 读取ApiConfig.ini配置文件

    host = cf.get("api-config", "host")  # 获取[api-config]中host对应的值
    port = cf.get("api-config", "port")  # 获取[api-config]中port对应的值
    appKey = cf.get("api-config", "appKey")  # 获取[api-config]中appKey对应的值
    appSecret = cf.get("api-config", "appSecret")  # 获取[api-config]中appSecret对应的值

    # Step2：设置接口地址及请求方式
    # artemis api
    content='artemis' #上下文 默认artemis
    api ='/api/resource/v2/regions/subRegions' # api 的url
    methon = 'POST' # POST 或 GET 请求

    # Step3：组装POST请求URL
    # Setting Url
    url = host +':' + port +'/' +  content +  api

    # Step4：获取安全认证的Headers
    # Setting Headers
    header_dict=Signature(appSecret,methon,appKey,content,api)

    # Step5：组装传入的Json
    # Setting JSON Body
    #payload = { "pageNo": 1, "pageSize": 1000 }
    payload={
        "parentIndexCode": regionIndex,
        "resourceType": "camera",
        "pageNo": 1,
        "pageSize": 1000,
        "cascadeFlag": 0
    }


    # Step6：发起POST请求
    # Make the requests
    r = requests.post(url, headers=header_dict, json=payload,verify=False)

    #Step7：解析请求响应
    # Check the response
    # print("status_code:",r.status_code)
    # print("content:",r.content.decode('utf-8'))
    return r.content.decode('utf-8')

aa={
    "code": "0",
    "msg": "SUCCESS",
    "data": {
        "total": 18,
        "pageNo": 1,
        "pageSize": 1,
        "list": [
            {
                "indexCode": "6e74e23d-8e4f-475d-a8b6-5f44e7161ac3",
                "name": "测试2",
                "parentIndexCode": "root000000",
                "catalogType": 10,
                "externalIndexCode": "11010508582160000029",
                "sort": 2,
                "regionPath": "@root000000@6e74e23d-8e4f-475d-a8b6-5f44e7161ac3@",
                "createTime": "2019-07-16T09:50:50.308+08:00",
                "updateTime": "2019-07-31T14:34:39.272+08:00",
                "available": "true",
                "cascadeCode": "0",
                "cascadeType": 0,
                "leaf": "true"
            }
        ]
    }
}

# def getLastRegion(regionindex):
#     preRegion=regionindex
#     region=getSubRegion(regionindex)
#     region=json.loads(region)
#     if(region["data"]["total"]>0):
#         for i in region["data"]["list"]:
#             regionindex=i["indexCode"]
#             if getSubRegion(i["indexCode"])["data"]["total"]==0:
#                 print(preRegion)
#
#             getLastRegion(getSubRegion(regionindex))
    # preRegion=regionindex
    # region=getSubRegion(regionindex)
    # # region=json.load(region)
    #
    # region=json.loads(region)
    # print(region)
    # if(region["data"]["total"]==0):
    #     return preRegion
    # #判断是否获取成功
    # else:
    #     for i in region["data"]["list"]:
    #         return getLastRegion(i["indexCode"])
    # # else:
    # #     print("preregion:", preRegion)
    # # getCameraIndex(preRegion)

#获取最下级区域列表
def getLastRegion(regionIndex):
    indexList = []
    root = regionIndex
    sub = getSubRegion(root)
    sub = json.loads(sub)
    if sub["data"]["total"] > 0:
        for i in sub["data"]["list"]:
            nextSub = getSubRegion(i["indexCode"])
            nextsub = json.loads(nextSub)
            if nextsub["data"]["total"] > 0:
                for j in nextsub["data"]["list"]:
                    nextSub1 = getSubRegion(j["indexCode"])
                    nextsub1 = json.loads(nextSub1)
                    if nextsub1["data"]["total"] > 0:
                        for k in nextsub1["data"]["list"]:
                            nextSub2 = getSubRegion(k["indexCode"])
                            nextsub2 = json.loads(nextSub2)
                            if nextsub2["data"]["total"] > 0:
                                for l in nextsub2["data"]["list"]:
                                    nextSub3 = getSubRegion(k["indexCode"])
                                    nextsub3 = json.loads(nextSub3)
                                    indexList.append(l["indexCode"])
                            else:
                                indexList.append(k["indexCode"])



                    else:
                        indexList.append(j["indexCode"])
            else:
                indexList.append(i["indexCode"])
    else:
        indexList.append(root)

    indexList = list(set(indexList))
    return indexList

if __name__ == '__main__':
    #aa=getLastRegion("37250000412169000001")

    # indexList=[]
    # root="root000000"
    # sub=getSubRegion(root)
    # sub=json.loads(sub)
    # print("sub:",sub)
    # if sub["data"]["total"] > 0:
    #     for i in sub["data"]["list"]:
    #         nextSub=getSubRegion(i["indexCode"])
    #         nextsub = json.loads(nextSub)
    #         if nextsub["data"]["total"] > 0:
    #             for j in nextsub["data"]["list"]:
    #                 nextSub1 = getSubRegion(j["indexCode"])
    #                 nextsub1 = json.loads(nextSub1)
    #                 if nextsub1["data"]["total"] > 0:
    #                     for k in nextsub1["data"]["list"]:
    #                         print(k["indexCode"])
    #                         nextSub2 = getSubRegion(k["indexCode"])
    #                         nextsub2 = json.loads(nextSub2)
    #                         if nextsub2["data"]["total"] > 0:
    #                             for l in nextsub2["data"]["list"]:
    #                                 nextSub3 = getSubRegion(k["indexCode"])
    #                                 nextsub3 = json.loads(nextSub3)
    #                                 indexList.append(l["indexCode"])
    #                         else:
    #                             indexList.append(k["indexCode"])
    #
    #
    #
    #                 else:
    #                     indexList.append(j["indexCode"])
    #         else:
    #             indexList.append(i["indexCode"])
    # else:
    #     indexList.append(root)
    #
    # indexList=list(set(indexList))
    # print(len(indexList))
    # print(indexList)
    print(getSubRegion("root000000"))