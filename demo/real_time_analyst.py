# -*- coding:utf-8 -*-
import csv
import datetime
import json
import re
import time
from urllib import request

import dateutil.parser
import requests
import sys

FILE_PATH = 'reuters_news.csv'
MONGO_URL = None
MONGO_USERNAME = None
MONGO_PASSWORD = None
HEADLINE_URL = 'http://mobile.reuters.com/assets/jsonHeadlines?channel=113&limit=5'
NEWS_URL = 'http://www.reuters.com/article/json/data-id'
_URL = "http://localhost:8000/predict"


def headline_url(last_time):
    if last_time == '':
        return HEADLINE_URL
    else:
        return HEADLINE_URL + '&endTime=' + last_time


def news_url(news_id):
    return NEWS_URL + news_id


def dump_news(news, csv_writer):
    # TODO visualize write news
    news_str = list(map(lambda x: str(x), news.values()))
    csv_writer.writerow(news_str)


res = request.urlopen(headline_url(''))
headlines = json.loads(res.read().decode())['headlines']

lastTime = headlines[-1]['dateMillis']
with open(FILE_PATH, 'a+') as f:
    writer = csv.writer(f)
    while lastTime != None:
        saved = 0
        for headline in headlines:
            try:
                news_res = request.urlopen(news_url(headline['id']))
                news_json = json.loads(news_res.read().decode())['story']
                content = re.sub(r'</?\w+[^>]*>', ' ', news_json['body']).replace('\n', ' ')
                published = dateutil.parser.parse(str(datetime.datetime.fromtimestamp(news_json['published']).date()))
                title = news_json['headline']
                source = NEWS_URL + headline['id']
                news = {'_id': headline['id'], 'title': title, 'date': str(published), 'content': content, 'url': source,
                        'whole_data': title + " " + content}
                try:
                    res = requests.post(
                        _URL,
                        json=news
                    )
                except requests.exceptions.RequestException:
                    print("ERROR: Request error, did you start model Serving?")
                    sys.exit()
                response = json.loads(res.content.decode("utf-8"))
                if response == {'Company': 0}:
                    continue

                print(response)
                news.update(response)
                dump_news(news, writer)
                lastTime = headline['dateMillis']
                saved += 1
            except Exception as e:
                lastTime = headline['dateMillis']
                print(e)
                continue

        print('Crawled at %s = %s saved %d' % (
            headlines[-1]['dateMillis'], str(datetime.datetime.fromtimestamp(int(headlines[-1]['dateMillis']) / 1000)),
            saved))
        time.sleep(5)
        res = request.urlopen(headline_url(lastTime))
        headlines = json.loads(res.read().decode())['headlines']

