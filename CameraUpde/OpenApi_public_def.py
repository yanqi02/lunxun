# -*- coding: UTF-8 -*-
import hashlib
import hmac
import base64
import time
import uuid 


def Signature(secret,methon,appKey,artemis,api):

    ## Timestamp
    t = time.time()
    nowTime = lambda:int(round(t * 1000))
    timestamp=nowTime()
    timestamp=str(timestamp)
    # uuid
    nonce= str(uuid.uuid1())
    #signature    
    secret=str(secret).encode('utf-8')
    message = str(methon+'\n*/*\napplication/json\nx-ca-key:'+appKey+'\nx-ca-nonce:'+nonce+'\nx-ca-timestamp:'+timestamp+'\n/'+artemis+api).encode('utf-8')
    signature = base64.b64encode(hmac.new(secret, message, digestmod=hashlib.sha256).digest())
#    print(signature)
   #header
    header_dict = dict()
    header_dict['Accept'] = '*/*'
    header_dict['Content-Type'] = 'application/json'
    header_dict['X-Ca-Key'] = appKey
    header_dict['X-Ca-Signature'] = signature
    header_dict['X-Ca-timestamp'] = timestamp
    header_dict['X-Ca-nonce'] = nonce
    header_dict['X-Ca-Signature-Headers'] = 'x-ca-key,x-ca-nonce,x-ca-timestamp'
#    print (header_dict)

    return header_dict

