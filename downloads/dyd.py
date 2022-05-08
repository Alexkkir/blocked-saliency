# download file from yandex disk
#example usage: ./dyd.py https://disk.yandex.ru/i/48HGhK4qSrImBw

import os, sys, json
import urllib.parse as ul
import sys

download_url = sys.argv[1]
folder = '.'

base_url = 'https://cloud-api.yandex.net:443/v1/disk/public/resources/download?public_key='
url = ul.quote_plus(download_url)
res = os.popen('wget -qO - {}{}'.format(base_url, url)).read()
json_res = json.loads(res)
filename = ul.parse_qs(ul.urlparse(json_res['href']).query)['filename'][0]
os.system("wget '{}' -P '{}' -O '{}'".format(json_res['href'], folder, filename))
