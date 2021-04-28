import requests
import pprint

sentences = ['I hate the man', 'The US fears Chinese politics']

res = requests.post('http://127.0.0.1:6001/opinion/mine', json={'sentences': sentences})

pprint.pprint(res.json())