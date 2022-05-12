import requests
import configparser
from RTSP.OpenApi_public_def import Signature
import json

'''

'''
def getCameraInfo(regionIndexCode):
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
    api = '/api/resource/v1/regions/regionIndexCode/cameras'  # api 的url
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
    "regionIndexCode": regionIndexCode,
    "pageNo": 1,
    "pageSize": 1000
}


    # Step6：发起POST请求
    # Make the requests
    r = requests.post(url, headers=header_dict, json=payload, verify=False)

    # Step7：解析请求响应
    return r.content.decode('utf-8')
#根据区域获取摄像头编号

def getCameraIndex(regionIndex):
    camreaList=[]
    for i in regionIndex:
        camreaInfo=getCameraInfo(i)
        camreaInfo=json.loads(camreaInfo)
        for j in camreaInfo["data"]["list"]:
            camreaList.append(j["cameraIndexCode"])
    return camreaList
def getCameraIndexAndName(regionIndex):
    camreaDict={}
    for i in regionIndex:
        camreaInfo=getCameraInfo(i)
        camreaInfo=json.loads(camreaInfo)
        for j in camreaInfo["data"]["list"]:
            camreaDict[j["cameraName"]]=j["cameraIndexCode"]
    return camreaDict
if __name__ == '__main__':
    aa=['37172000412169050862', '37150300412169100002', '37170303412169276310', '37172000412169279622', '37160700412169016972', '37150707412169085461', '37218800412169900986', '160407030045225190', '37050248005010000017', '37150100012169670696', '37218800412169289012', '37150000412169123123', '37160700412169573054', '37170216000000000000', '34000000582160000054', '37170201000000000000', '37170218000000000000', '65432101582160000090', '37050278005010000036', '37171000412169685695', '200827045174907690', '379654', '37150501412169000000', '37161200412169539744', '37160912412169549247', '160407035246492090', '37170207000000000000', '37050074472160000083', '37150300412169009523', '37050217582160000008', '37150707412169547157', '190802103995864390', '37050365582160000007', '37150707412169549629', '37171000412169811536', '37160600412169237560', '37150800412169682965', '37161200412169039417', '210511114859239090', '37250801412169000000', '190612113266731090', '37218800412169949782', '37150707412169536261', '37172000412169626196', '37018100412169018753', '37161200412169070272', '160407035348415290', '37050074472160000081', '37160700412169300385', '160407035235200890', '37050258005010000025', '37160700412169494069', '37160912412169311626', '37018100412169719525', '37160700412169968223', '160407035223559690', '37170303412169069691', '37050265005010000034', '37150707412169810276', '379656', '37170205000000000000', '37172000412169857162', '37161100412169639420', '37161100412169131790', '37150707412169170558', '37170303412169325556', '37218800412169576080', '37050217582160000022', '37050260005010000027', '37160700412169088475', '37160700412169073453', '37050074472160000085', '160420080866622290', '37160600412169560510', '37170203000000000000', '37161200412169622778', '37161200412169003207', '37050217582160000006', '37160800412169564496', '37170303412169520495', '200111094096370090', '37171000412169301013', '37161100412169283664', '160407035256926790', '37150707412169037649', '37150300412169449714', '37170102412169000000', '37150505412169000000', '37170800412169531478', '37050251005010000020', '37150100012169969678', '37170109412169000000', '37050217582160000007', '37018100412169575161', '37150100012169132276', '37250401412169000000', '37150300412169019906', '37250501412169000000', '37150110', '37160600412169487612', '200710044934089790', '37161200412169523556', '37161100412169740930', '37161200412169537089', '37050261005010000028', '37161200412169511626', '37050259005010000026', '37160912412169409087', '37250200412169000000', '160904122341609390', '37218800412169585128', '37170800412169615345', '37172000412169291471', '37050217582160000019', '37160700412169871651', '37160700412169724301', '37050242005010000009', '37150504412169000000', '379660', '37160000412161200000', '160407030042205490', '161207074869741690', '161018021555005490', '37050074472160000082', '37170101412169000000', '37172000412169478381', '37018100412169517314', '37018100412169649421', '37150503412169000000', '37158700412169000001', '160407035229973890', '37168811412169234567', '37018100412169107571', '37160700412169729480', '160420074551997890', '37170303412169654472', '37018100412169696770', '190102044992393990', '37160610412169000010', '200610092620550590', '37161100412169439050', '37250602412169000000', '37050262005010000029', '37150300412169291383', '37050241005010000008', '37160700412169077629', '37160700412169697120', '37018100412169301168', '37050074472160000086', '160407030183995790', '160407035388416990', '37170219000000000000', '37050247005010000016', '37172000412169401554', '37150506412169000000', '37161100412169238007', '37050074472160000078', '37218800412169878295', '37161200412169270207', '37150707412169625683', '37050254005010000012', '37161200412169710849', '37250902412169000000', '37218800412169646362', '37050217582160000005', '37161200412169526947', '37160700412169724334', '37170303412169036745', '37050365582160000002', '37150800412169850370', '37170800412169053728', '37161200412169291072', '37160700412169438212', '37050250005010000019', '37160700412169246560', '37218800412169648989', '37150100012169222708', '37172000412169804658', '379657', '37161200412169155639', '37160700412169535002', '160407035140161090', '37018100412169404894', '160420075361605890', '160420021495988790', '37160700412169410018', '37160800412169350504', '191212095675888490', '37150300412169476843', '37161200412169029330', '37160800412169621283', '37170209000000000000', '37250302412169000000', '37218800412169627166', '37170800412169091847', '37160707412169414475', '37160700412169105699', '37018100412169180049', '37160700412169255911', '37160707412169700619', '37170212000000000000', '37161100412169601825', '37018100412169398199', '37150100412169000104', '37018100412169945685', '379653', '37018100412169443044', '37150707412169172648', '379658', '37150300412169994072', '200901030740441390', '37250601412169000000', '37050243005010000010', '37160912412169644681', '37050217582160000004', '37160700412169000924', '37172000412169637928', '37170800412169264003', '37150001412169134024', '37160700412169978551', '37050074472160000084', '37250301412169000001', '37170105412169000000', '37161100412169831509', '37170217000000000000', '37050252005010000021', '37160912412169197479', '37170208000000000000', '37160700412169914557', '37050245005010000014', '37160809412169000008', '37171000412169075622', '37018100412169330249', '37170800412169892531', '37170103412169000000', '37160700412169464047', '37160800412169782450', '37684802442160382746', '37160800412169181076', '37172000412169162649', '37150502412169000000', '37170204000000000000', '37172000412169149215', '37170108412169000000', '37160700412169245822', '37161200412169695223', '37170303412169710701', '37170303412169143908', '37170107412169000000', '37150300412169825948', '37160912412169183314', '37170800412169555940', '37170210000000000000', '37150800412169158074', '37160800412169748572', '37050217582160000023', '37050365582160000008', '170105043750057090', '37150000412169120000', '37150707412169156512', '37050247005010000015', '37160700412169358445', '37171000412169025473', '37170206000000000000', '37170303412169529145', '37161100412169780586', '37150300412169750956', '37250501412169000001', '37161200412169640248', '37160010412169100180', '37250904412169000000', '37170303412169507990', '37170211000000000000', '37160600412169668082', '160407030023575690', '37250100412169000000', '37170104412169000000', '37171000412169068143', '37018100412169971983', '37150707412169526942', '37150509412169000000', '37218800412169825303', '37161100412169280099', '37150400412169000000', '37170800412169924185', '37160700412169207647', '37018100412169747334', '37150707412169731922', '160503095739787690', '37018100412169877380', '37161200412169678547', '37161100412169368700', '210422125481673090', '160407035395610890', '37160700412169297983', '37050074472160000077', '37160700412169164804', '37161200412169027524', '37160912412169073654', '37050074472160000080', '37160700412169655198', '160412045086658390', '37160700412169890987', '37172000412169554780', '37218800412169555692', '37250903412169000000', '37170202000000000000', '37150100012169008688', '37160700412169734163', '37050257005010000024', '37018100412169666727', '37018100412169304469', '37018100412169549070', '37160912412169754956', '37160800412169711755', '37170215000000000000', '37018100412169422789', '37170214000000000000', '37171000412169273288', '37018100412169981376', '37170110412169000000', '37170303412169302343', '37160700412169124618', '37160700412169700353', '37050074472160000079', '37218800412169340834', '37160800412169686773', '37160700412169755850', '37250901412169000000', '37160505412169000023', '37161200412169438519', '37150800412169387507', '37250905412169000000', '37150800412169113307', '37160700412169120106', '160420080166867790', '37150300412169538910', '37250603412169000000', '37050246005010000011', '37018100412169419076', '37161100412169153070', '37170303412169185517', '37218800412169311247', '37050249005010000018', '379659', '37160912412169320279', '37170106412169000000', '37172000412169573384', '37150800412169323343', '37161100412169183484', '160407035323496790', '37160700412169186343', '37170303412169898688', '37050217582160000003', '37301001412169100010', '160412091728162790', '180425094128431790', '37170800412169065480', '37160700412169001021', '37170303412169938575', '37160700412169515866', '37018100412169846757', '37170213000000000000', '37050263005010000031', '37171001412169000011', '37171000412169481323', '37160700412169139474', '37170303412169508338']
    aaa=getCameraIndex(aa)
    print(len(aaa))