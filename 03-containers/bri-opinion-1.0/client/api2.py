import requests
import pprint

sentences = ["One of the first famous examples is the Tanzania - Zambia Railway built between 1970 and 1975", 
"Chinese financing to Sri Lanka from 2001 to 2017 carry only 2 percent interest."]

res = requests.post('http://127.0.0.1:6000/event/detection', json={'sentences': sentences, 'lang': 'eng', 'format': 'normal'})
pprint.pprint(res.json())