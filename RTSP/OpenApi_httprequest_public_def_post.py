# -*- coding: UTF-8 -*-
import requests
import configparser
from OpenApi_public_def import Signature

# Step1：从配置文件获取host地址、端口号、appKey和appSecret
# api config
cf = configparser.ConfigParser()
cf.read(".\ApiConfig.ini")  # 读取ApiConfig.ini配置文件

host = cf.get("api-config", "host")  # 获取[api-config]中host对应的值
#print(host)

port = cf.get("api-config", "port")  # 获取[api-config]中port对应的值
#print(port)

appKey = cf.get("api-config", "appKey")  # 获取[api-config]中appKey对应的值
#print(appKey)

appSecret = cf.get("api-config", "appSecret")  # 获取[api-config]中appSecret对应的值
#print(appSecret)

# Step2：设置接口地址及请求方式
# artemis api
content='artemis' #上下文 默认artemis
api ='/api/resource/v1/org/advance/orgList' # api 的url 
methon = 'POST' # POST 或 GET 请求

# Step3：组装POST请求URL
# Setting Url
url = host +':' + port +'/' +  content +  api 
print('requesturl:'+ url)

# Step4：获取安全认证的Headers
# Setting Headers
header_dict=Signature(appSecret,methon,appKey,content,api)

# Step5：组装传入的Json
# Setting JSON Body
payload = { "pageNo": 1, "pageSize": 1000 }
print(payload)

# Step6：发起POST请求
# Make the requests
r = requests.post(url, headers=header_dict, json=payload,verify=False)

#Step7：解析请求响应
# Check the response
print(r.status_code)
print(r.content.decode('utf-8'))
