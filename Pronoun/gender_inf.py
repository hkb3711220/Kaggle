import pandas as pd
import os
os.chdir(os.path.dirname(__file__))
from bs4 import BeautifulSoup
import requests
import re

table = pd.read_table('./test_stage_1.tsv')
table.head()

class get(object):

    def __init__(self):

        self.HEADER = {'User-Agent': 'Mozilla/5.0'}

    def response(self, url):

        try:
            response = requests.get(url, headers=self.HEADER)
            response.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print("Http Error:", errh)
        except requests.exceptions.ConnectionError as errc:
            print("Error Connecting:", errc)
        except requests.exceptions.Timeout as errt:
            print("Timeout Error:", errt)
        except requests.exceptions.RequestException as err:
            print("OOps: Something Else", err)

        soup = BeautifulSoup(response.text, 'lxml')

        return soup


def search(soup, target):

    lis = soup.select("li")
    ps  = soup.select("p")

    for p in ps:
        if re.search(target, p.text): return p.text
    # if p does not exist. target
    for li in lis:
        if re.search(target, li.text): return li.text
