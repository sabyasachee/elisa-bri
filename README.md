# Opinion Mining

#### Requirements

Tested with Python 3.6+

##### Packages
```
spacy>=3.0
```

##### Models

Download spacy models:
```
python -m spacy.download en_core_web_sm
python -m spacy.download zh_core_web_sm
python -m spacy.download es_core_news_sm
```

#### Preprocessing TAC KBP BeST data (2016-2017)

1. Download the TAC KBP BeST data from LDC (LDC2017E80, LDC2016E114), and point `BEST_2017_DIR` and `BEST_2016_DIR` in _config.py_ to their folders.

2. `python preprocess_best.py` preprocesses the BeST data into json format in `RESULTS_DIR`, defined in _config.py_. You can select between 2016 and 2017 data by changing the `best_dir` variable of _preprocess_best.py_.