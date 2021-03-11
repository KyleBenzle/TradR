from typing import get_type_hints
import requests,time
from datetime import datetime, timedelta
from urllib.parse import urlencode
# timing = "&start=2018-04-14T00%3A00%3A00Z&end=2014T00%3A00%3A00Z"
api = "0bf0bfc5ac55e807e6448ac4b683ef8b"
url = "https://api.nomics.com/v1/volume/history?key="+api


cur_time = datetime.now() #- timedelta(hours=1)
cur_time_formatted = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
last_hour_date_time = cur_time - timedelta(hours=1)
last_hour_date_time_formatted = last_hour_date_time.strftime('%Y-%m-%dT%H:%M:%SZ')
currencies = ["BTC","ETH","BCH","XMR","DASH"]

volumes = []
def getVolumes():

    for cur in currencies:
        params = {
            "start" : last_hour_date_time_formatted,
            "end" : cur_time_formatted,
            "convert" : cur
        }

        url_param  = urlencode(params)

        new_url = url+"&"+url_param
        time.sleep(1)
        print(new_url)
        try: 
            data = requests.get(new_url).json()
            print(data)
            val = data[-1]['volume']
            print(val)
            volumes.append(val)
        except Exception as e:
            print(e)
            volumes.append("None")
        # print(data)
    return volumes