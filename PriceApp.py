  
 #This example uses Python 2.7 and the python-request library.

import datetime
from requests import Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json,csv,time
from nomics import getVolumes
import os.path

url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
parameters = {
  'start':'1',
  'limit':'5000',
  'convert':'USD'
}
headers = {
  'Accepts': 'application/json',
  # https://pro.coinmarketcap.com/account/
  # you can get that from here
  'X-CMC_PRO_API_KEY': 'abf52351-f0cb-4233-849f-bf948c944ace',
}

session = Session()
session.headers.update(headers)
# it will extract this coin prices
# BTC,ETH,BCH,XMR,DASH,FIL,BAT,ZRX,REP,KNC
l = ["BTC","ETH","BCH","XMR","DASH"]

# while True:

print("\n\n")
print("="*50)
print("info : starting...")

if os.path.isfile('PriceData.csv'):
    print ("info : File already exist")
else:
    with open("PriceData.csv","w+",newline='', encoding='utf-8') as f:
      writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      writer.writerow(['Time',"BTC","ETH","BCH","XMR","DASH","BTC_volume","ETH_volume","BCH_volume","XMR_volume","DASH_volume"])
    print ("info : File not exist... will create new file")

rates = []

print("info : Wait till data get loaded in the file...")

try:
  response = session.get(url, params=parameters)
  data = json.loads(response.text)
  data = data['data']
  rates.append(datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))
  for d in l:
    for a in data:
      # with open('file.txt',"a") as f:
      #   f.write(a['symbol']+"\n")
      if(a['symbol'] == d ):
        rate = a['quote']['USD']['price']
        rates.append(rate)
  coin_volumes = getVolumes()
  # coin_volumes = set(coin_volumes)
  for coin in coin_volumes:
    rates.append(coin)
  with open('PriceData.csv',"a+",newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(rates)
  rates = []
  coin_volumes= []

except (ConnectionError, Timeout, TooManyRedirects) as e:
  print(e)


print("success: Complete...")



  # break
  # time.sleep(3)