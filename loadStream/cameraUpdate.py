from RTSP.getCameraIndex import getCameraIndex
from RTSP.getSubRegions import getLastRegion

import requests
import configparser
from RTSP.OpenApi_public_def import Signature
import json

def getCameraByRegion(regionIndex):
    allCameras=getCameraIndex(getLastRegion(regionIndex))
    return allCameras
if __name__ == '__main__':


    print(getCameraByRegion("37"))
