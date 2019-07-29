from urllib import request
import dateutil
from datetime import datetime
import re
import pandas as pd
from pymongo import MongoClient
import nltk
import numpy as np
import string
import functools
import spacy
from sklearn.preprocessing import scale
from tqdm import tqdm
nlp = spacy.load('en', disable=["tagger"])

MONGO_URL = "mongodb://localhost:27017/"
MONGO_USERNAME = None
MONGO_PASSWORD = None

client = MongoClient(MONGO_URL)
db = client.stockdb
stock_coll = db.stockcoll
print(stock_coll.count())
news_collect = db.news_collect
samples = list(news_collect.find())



def tokenize_remove_stopwords_extract_companies_with_spacy(text, sample_date, companies):
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.append('would')
    stopwords.append('kmh')
    stopwords.append('mph')
    stopwords.append('u')
    stopwords.extend(list(string.ascii_lowercase))
    stop_symbols = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG',
                    'SEP', 'OCT', 'NOV', 'DEC', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
    processed_data = []
    regex = re.compile(r'[^A-Za-z-]')
    doc = nlp(text)
    sentences = list(doc.sents)
    for sentence in sentences:
        tokens = list(map(str, sentence))
        complete_sentence = str(sentence)
        sent_doc = nlp(complete_sentence)
        entities = list(map(str, sent_doc.ents))
        for company in companies:
            if company[1] in entities or company[2] in complete_sentence or company[0] in entities:
                future_price_data = list(stock_coll.find(
                    {'symbol': company[0], 'date': {'$gte': sample_date}}).limit(2))
                past_price_data = pd.DataFrame(list(stock_coll.find(
                    {'symbol': company[0], 'date': {'$lte': sample_date}}).sort('date', -1).limit(7)))
                if len(past_price_data) != 7:
                    continue
                past_price_data = scale(
                    past_price_data['adj_close'].values[0:-1]-past_price_data['adj_close'].values[1:])
                if len(future_price_data) < 2:
                    continue
                if (future_price_data[0]['date']-sample_date).days > 3:
                    continue
                # get the open price of stock minus the end day price of stock
                price_label = np.sign(future_price_data[1]['adj_close']-future_price_data[0]['adj_close'])
                processed_data.append((complete_sentence, tokens, sent_doc,
                                       company[0], company[1], company[2], price_label, past_price_data, sample_date))
    return processed_data


def main():

    companies = pd.read_csv(
        'https://datahub.io/core/s-and-p-500-companies-financials/r/constituents-financials.csv')
    companies.columns = list(
        map(lambda x: x.strip().lower(), companies.columns))
    # companies=companies[companies.symbol.isin(['GOOGL','IBM','ORCL','AAPL','YHOO','FB'])]
    companies.index = companies['symbol']
    companies = companies[['symbol', 'name', 'sector']]
    company_names = companies['name'].values
    company_symbols = companies['symbol'].values
    company_info = companies[['symbol', 'name', 'name']].values

    stop_company_name = ['&', 'the', 'company', 'inc', 'inc.', 'plc',
                         'corp', 'corp.', 'co', 'co.', 'worldwide', 'corporation', 'group', '']
    # stop_company_name=[]
    splitted_companies = list(map(lambda x: ([x[0]]+[x[1]]+list(filter(
        lambda y: y.lower() not in stop_company_name, x[2].split(' ')))), company_info))
    splitted_companies = list(map(lambda x: [x[0]]+[x[1]]+[re.sub(pattern='[^a-zA-Z0-9\s-]',
                                                                  repl='', string=functools.reduce(lambda y, z:y+' '+z, x[2:]))], splitted_companies))

    processed_samples = []
    count = 0
    to_date = "2006-10-01"
    from_date = "2013-11-01"
    to_date = datetime.strptime(to_date, "%Y-%m-%d")
    from_date = datetime.strptime(from_date, "%Y-%m-%d")
    for sample in tqdm(samples):
        if from_date <= sample['date'] <= to_date:
            try:
                p_sample = tokenize_remove_stopwords_extract_companies_with_spacy(
                    str(sample['heading_news']) + " " + str(sample["body_news"]), sample['date'], splitted_companies)
                if len(p_sample) == 0:
                    continue
                p_sample = np.array(p_sample)
                processed_samples.extend(p_sample)
                if len(processed_samples) % 5000 == 0:
                    temp_sample = np.array(processed_samples)
                    print("save data checkpoint")
                    np.save("body_data_point_date2", temp_sample)
            except Exception as e:
                print("wrong: ", e)
                continue
        count+=1
        
    processed_samples = np.array(processed_samples)
    np.save("body_data_point", processed_samples)   
    print("Done saving")
    df = pd.DataFrame(processed_samples,
                      columns=["complete_sentence", "tokens", "sent_doc", "company_1", "company_2", "company_3", "price_label", "past_price_data", "sample_date"])
    df.to_csv("body_data.csv")

if __name__ == "__main__":
    main()
