from alpha_vantage.cryptocurrencies import CryptoCurrencies
import json

API_key = '010EQDKOC8H57DUF'
cur = "BTC"
period = "daily"
toCur = "USD"

#ts = TimeSeries(key = API_key, output_format = "pandas")
cc = CryptoCurrencies(API_key, output_format = "json")

#data = ts.get_intraday('AAPL',interval = "5min")
data = cc.get_digital_currency_exchange_rate(cur, toCur)

y = json.dumps(data)


print(y)
