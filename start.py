import pandas as pd
import numpy as np
from pandas.io.json import json_normalize

import requests
import json

import urllib


credentials = {'teamid' : 'Coffeeblack',
               'teampw' : '23e88f60c155804c79cfd1af83e47fc8'}


def getcontext_r(i, runid):

    baseurl = "http://krabspin.uci.ru.nl/getcontext.json/"
    payload = {'i' : i,
               'runid' : runid}

    payload.update(credentials)  # Add credentials
    r = requests.get(baseurl, params=payload)  # GET request

    return r


def create_df(runid):

    rs = []  # responses
    for i in range(1000):
        rs.append(getcontext_r(i, runid).json())

    df = json_normalize(rs)
    df.columns = ['Age', 'Agent', 'ID', 'Language', 'Referer']
    df.to_csv(str(runid) + ".csv", index=False)
    
    return df


def proposepage(i, runid, header, adtype, color, productid, price):

    payload = {'i' : i,
               'runid': runid,
               'price': price,
               'header': header,
               'adtype': adtype,
               'color': color,
               'productid': productid,
               'price': price}

    payload.update(credentials)  # Add credentials

    # Propose page and get JSON answer
    r = requests.get("http://krabspin.uci.ru.nl/proposePage.json/", params=payload)
    print(r.text)
