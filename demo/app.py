import argparse
import datetime
import json
import os
import re
import sys

import urllib

import dateutil.parser
import requests
from flask import Flask, flash, request, redirect, render_template, Response

app = Flask(__name__)
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
current_dir = os.path.dirname("__file__")
current_dir = current_dir if current_dir is not '' else '.'

parser = argparse.ArgumentParser()
parser.add_argument("--host")
parser.add_argument("--port")
args = parser.parse_args()
_URL = "http://localhost:8000/predict"

app.config['SECRET_KEY'] = 'f3cfe9ed8fae309f02079dbf'


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/convertfromlink', methods=['GET'])
def convertIndex():
    return render_template('upload_link.html')


@app.route('/convertfromlink', methods=['POST'])
def convert():
    text = request.form['text']
    print(text)
    if "reuters" not in text:
        flash('Not valid link. Only support reuters ')
        return redirect(request.url)

    news_url = text.replace("article/", "article/json/")
    news_res = urllib.request.urlopen(news_url)
    news_json = json.loads(news_res.read().decode())['story']
    content = re.sub(r'</?\w+[^>]*>', ' ', news_json['body']).replace('\n', ' ')
    published = dateutil.parser.parse(str(datetime.datetime.fromtimestamp(news_json['published']).date()))
    title = news_json['headline']
    source = news_url
    id = news_url.split("-")[-1]
    news = {'_id': id, 'title': title, 'date': str(published), 'content': content, 'url': source,
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
    print(response)
    news.update(response)
    if news:
        return Response(json.dumps(news), 200, mimetype="application/json")

    else:
        return render_template('vietnameseError.html')

@app.route('/convertfromtext', methods=['GET'])
def convert_text():
    return render_template('upload_text.html')

@app.route('/convertfromtext', methods=['POST'])
def convert_from_text():
    text = request.form['text']
    if not text:
        flash('Text none')
        return redirect(request.url)

    news = {'date': "now", 'whole_data': text}
    try:
        res = requests.post(
            _URL,
            json=news
        )
    except requests.exceptions.RequestException:
        print("ERROR: Request error, did you start model Serving?")
        sys.exit()
    response = json.loads(res.content)
    news.update(response)
    if news:
        return Response(json.dumps(news), 200, mimetype="application/json")

    else:
        return render_template('error.html')


if __name__ == "__main__":
    app.run(host=args.host, port=args.port)
