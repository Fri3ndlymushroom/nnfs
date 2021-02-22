from bs4 import BeautifulSoup
import requests
import pandas
import json
import time





def makeFloat(worth):
    x = ""
    for i in worth:
        if i == "$":
            x = x
        else:
            x = x + i
    return float(x)
    


controll = 0

lastWorth = 0
worth = 0


X = []
y = []
for x in range(100):
    courses = []
    while (len(courses) < 4):
        cmc = requests.get(
            'https://www.livecoinwatch.com/price/Bitcoin-BTC')
        soup = BeautifulSoup(cmc.content, 'html.parser')
        lastWorth = soup.find("span", {"price"}).get_text()

        print(x, makeFloat(lastWorth))

        time.sleep(900)
        cmc = requests.get(
            'https://www.livecoinwatch.com/price/Bitcoin-BTC')
        soup = BeautifulSoup(cmc.content, 'html.parser')
        worth = soup.find("span", {"price"}).get_text()

        diff = makeFloat(lastWorth) - makeFloat(worth)

        indicator = "/"
        if diff > 0:
            indicator = "/"
        else:
            indicator = chr(92)

        print(diff, indicator)


        courses.append(diff)

        


    time.sleep(3600)
    cmc = requests.get(
            'https://www.livecoinwatch.com/price/Bitcoin-BTC')
    soup = BeautifulSoup(cmc.content, 'html.parser')
    controll = soup.find("span", {"price"}).get_text()

    


    y = makeFloat(worth) - makeFloat(controll)

    X.append(courses)
    y.append(controll)
    print("---------------------------------")
    print("|", r, "|", courses, "|", controll, "|")
    print("---------------------------------")
    r += 1


print(X, y)







