import requests
api_key = 'cd83fd1ed69545d:ivc5icl7su21s5m'
url = f'https://api.tradingeconomics.com/calendar?c={api_key}'
data = requests.get(url).json()
print(data)