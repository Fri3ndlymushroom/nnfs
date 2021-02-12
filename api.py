from bs4 import BeautifulSoup
import requests
import pandas
import json
import time





worthArray = []
lastWorth = 0



while (len(worthArray) < 20):
    cmc = requests.get(
    'https://coinkurs.com/bitcoin-kurs-euro.html')
    soup = BeautifulSoup(cmc.content, 'html.parser')
    worth = soup.find("div", {"BTCEUR"}).get_text()

    worth2 = ""

    x = 0
    for i in worth:
        if x != 2:
            worth2 = worth2 + i
        x += 1

    worth2 = float(worth2)

    print("---", len(worthArray))
   

    if lastWorth != worth2:
        print("------")
        print(worth2)
        worthArray.append(worth2)

    lastWorth = worth2
    time.sleep(2)





X = []
y = []

i = 0
while i < 10:
    x = 0
    XHolder = []
    while x < 10:

        XHolder.append(round(worthArray[len(worthArray)-2 - x - i] -
                       worthArray[len(worthArray)-3 - x - i], 2))

        x += 1
    X.append(XHolder)
    i += 1

    if worthArray[len(worthArray) - i] < worthArray[len(worthArray) - 1-i]:

        y.append(0)
    else:
        y.append(1)

worthArray[len(worthArray)-2 - 0 - 0] - worthArray[len(worthArray)-3 - 0 - 0]

print("----")
print(X, y)