import requests
import time
import json

from utils.prints import logMsg

import random

def requestJson(url, secret, timeout=30, retries=999999):
    cnt = 0
    while True:
        cnt += 1
        try:
            response = requests.get(url=url, headers={"secret": secret}, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            return data
        except Exception as error:
            if cnt > retries:
                raise error
            else:
                logMsg("Failed requestJson %s will retry soon." % url, error)
                time.sleep(timeout // 4 + random() * 20)


def requestBytes(url, secret, timeout=30, retries=999999):
    cnt = 0
    while True:
        cnt += 1
        try:
            response = requests.get(url=url, stream=True, headers={"secret": secret}, timeout=timeout)
            response.raise_for_status()
            return response.raw.data
        except Exception as error:
            if cnt > retries:
                raise error
            else:
                logMsg("Failed requestBytes %s will retry soon" % url, error)
                time.sleep(timeout // 4 + random() * 20)

def postJson(url, secret, data, timeout=30, retries=999999, getResponse=False):
    cnt = 0
    while True:
        cnt += 1
        try:
            dj = json.dumps(data)
            response = requests.post(url=url, data=dj, headers= {
                "secret": secret,
                "Content-Type": "application/json;charset=utf-8"
            })
            response.raise_for_status()
            if getResponse:
                return response.json()
            else:
                return
        except Exception as error:
            if cnt > retries:
                raise error
            else:
                logMsg("Failed postJson %s will retry soon" % url, error)
                time.sleep(timeout // 4 + random() * 20)

def postBytes(url, secret, data, timeout=30, retries=999999):
    cnt = 0
    while True:
        cnt += 1
        try:
            response = requests.post(url, data=data, headers={"secret": secret}, timeout=timeout)
            response.raise_for_status()
            return
        except Exception as error:
            if cnt > retries:
                raise error
            else:
                logMsg("Failed postBytes %s will retry soon" % url, error)
                time.sleep(timeout // 4 + random() * 20)

