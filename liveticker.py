from bs4 import BeautifulSoup
import requests
import pandas
import json
import time


worthArray = []
lastWorth = 0
X = []


while (len(worthArray) < 20):
    cmc = requests.get(
        'https://www.livecoinwatch.com/price/Bitcoin-BTC')
    soup = BeautifulSoup(cmc.content, 'html.parser')
    worth = soup.find("span", {"price"}).get_text()

    worth2 = ""

    x = 0
    for i in worth:
        if x == 0:
            x = x
        elif x == 0:
            x = x
        else:
            worth2 = worth2 + i
        x += 1

    worth2 = float(worth2)

    if lastWorth != worth2:
        print("[", len(worthArray), "] 1BTC = $", worth2)
        worthArray.append(worth2)

    lastWorth = worth2
    time.sleep(1)



print(worthArray)


controll = []
X = []
y = []

i = 0
while i < 10:
    x = 0
    XHolder = []
    controllHolder = []
    while x < 10:
        controllHolder.append(worthArray[len(worthArray)- 1 - x - i])
        XHolder.append(round(worthArray[len(worthArray)-2 - x - i] -
                       worthArray[len(worthArray)-3 - x - i], 2))

        x += 1
    X.append(XHolder)
    controll.append(controllHolder)
    i += 1

    if worthArray[len(worthArray) - i] < worthArray[len(worthArray) - 1-i]:

        y.append(0)
    else:
        y.append(1)

worthArray[len(worthArray)-2 - 0 - 0] - worthArray[len(worthArray)-3 - 0 - 0]


print(X, y)
print("[######################]")

r = 0
for i in X:
    #print(controll[r])
    print(i)
    print(y[r])
    print("[######################]")
    r += 1