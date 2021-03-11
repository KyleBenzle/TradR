#!/usr/bin/env python
# coding: utf-8

# In[289]:


# Run first time only

import re
import ast
import csv
import nltk
import warnings
import numpy as np
import pandas as pd
from ast import literal_eval
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from itertools import zip_longest
from nltk import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.impute import SimpleImputer
from nltk.tokenize import RegexpTokenizer
from gensim.utils import simple_preprocess
from sklearn import datasets, linear_model
from gensim.test.utils import common_texts
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import classification_report
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor





##############
# log file

import sys
old_stdout = sys.stdout

log_file = open("message.log","w")

sys.stdout = log_file

print ("this will be written to message.log")
# log file
############



#######
# Date time
import datetime

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))

#########


# Downloads
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Ignore the warnings
warnings.filterwarnings('ignore')


# In[290]:


# Read in scrapped data
df = pd.read_csv ('./ScrappedData/ScrappedReddit.csv')


PriceData = pd.read_csv ('./PriceData.csv', error_bad_lines=False)


PriceData = PriceData.iloc[:,:11]

# Read in buy prices
# buyPrice = pd.read_csv ('BuyPrice.csv')


PriceData.columns=['date', 'bitcoin_price', 'ethereum_price', 'btc_price', 'monero_price',
       'dashpay_price', 'bitcoin_volume', 'ethereum_volume', 'btc_volume',
       'monero_volume', 'dashpay_volume']


# Parse and merge data
PriceData["parsedDate"] = pd.to_datetime(PriceData["date"], infer_datetime_format=True)

PriceData["dateOnly"] = PriceData["parsedDate"].dt.date

PriceData["hourOnly"] = PriceData["parsedDate"].dt.hour


df["parsedDate"] = pd.to_datetime(df["Hour"], infer_datetime_format=True)
df["dateOnly"] = df["parsedDate"].dt.date
df["hourOnly"] = df["parsedDate"].dt.hour

# Merge on dates
mergedDf = df.merge(PriceData, how='left', on=['dateOnly', 'hourOnly'])

# Drop unused columns
mergedDf = mergedDf.drop(columns=['parsedDate_y'])
mergedDf = mergedDf.drop(columns=['parsedDate_x'])
# mergedDf = mergedDf.drop(columns=['date_x'])
# mergedDf = mergedDf.drop(columns=['date_y'])

# Merge the data and prices
mergedDf['bitcoin_pctChange'] = mergedDf['bitcoin_price'].pct_change().shift(-1)
mergedDf['btc_pctChange'] = mergedDf['btc_price'].pct_change().shift(-1)
mergedDf['ethereum_pctChange'] = mergedDf['ethereum_price'].pct_change().shift(-1)
mergedDf['monero_pctChange'] = mergedDf['monero_price'].pct_change().shift(-1)
mergedDf['dashpay_pctChange'] = mergedDf['dashpay_price'].pct_change().shift(-1)
mergedDf['ethfinance_pctChange'] = mergedDf['ethereum_price'].pct_change().shift(-1)
mergedDf['ethtrader_pctChange'] = mergedDf['ethereum_price'].pct_change().shift(-1)
mergedDf['xmrtrader_pctChange'] = mergedDf['monero_price'].pct_change().shift(-1)

# Copy prices for trader subs
mergedDf['xmrtrader_price'] = mergedDf['monero_price']
mergedDf['ethfinance_price'] = mergedDf['ethereum_price']
mergedDf['ethtrader_price'] = mergedDf['ethereum_price']

mergedDf['bitcoin_signal'] = (mergedDf['bitcoin_pctChange'] > 0).astype(int)
mergedDf['btc_signal'] = (mergedDf['btc_pctChange'] > 0).astype(int)
mergedDf['ethereum_signal'] = (mergedDf['ethereum_pctChange'] > 0).astype(int)
mergedDf['monero_signal'] = (mergedDf['monero_pctChange'] > 0).astype(int)
mergedDf['dashpay_signal'] = (mergedDf['dashpay_pctChange'] > 0).astype(int)
mergedDf['ethfinance_signal'] = (mergedDf['ethfinance_pctChange'] > 0).astype(int)
mergedDf['ethtrader_signal'] = (mergedDf['ethtrader_pctChange'] > 0).astype(int)
mergedDf['xmrtrader_signal'] = (mergedDf['xmrtrader_pctChange'] > 0).astype(int)


# Rename and drop unused columns
df = mergedDf.copy()
df = df.drop(columns=['dateOnly', 'hourOnly'])

# Function for extracting the comments
def extract_comment(df, coinComments):
    comment_post = [] # Opening empty list
    
    for i in range(len(df)):
        dictTest = df[coinComments][i]
        
        try:            
            dictTest = ast.literal_eval(dictTest)
            final_comment = "" 
            
            for element in dictTest:
                if element['comments'] != '0':
                    for comment in element['comments_on_post']:
                        final_comment += comment + " "                   
            if final_comment != "":
                comment_post.append(final_comment)        
            else:
                comment_post.append(np.nan)
        except:
            comment_post.append(np.nan)
    return comment_post


# Passing  the dataframes into the function
btcCommentsBlob = extract_comment(df, 'btc_comments')
bitcoinCommentsBlob = extract_comment(df, 'bitcoin_comments')
ethereumCommentsBlob = extract_comment(df, 'ethereum_comments')
moneroCommentsBlob = extract_comment(df, 'monero_comments')
dashpayCommentsBlob = extract_comment(df, 'dashpay_comments')
ethtraderCommentsBlob = extract_comment(df, 'ethtrader_comments')
ethfinanceCommentsBlob = extract_comment(df, 'ethfinance_comments')
xmrtraderCommentsBlob = extract_comment(df, 'xmrtrader_comments')


#Converting into dataframe
dfComments = pd.DataFrame({'btc_comments':btcCommentsBlob, 'bitcoin_comments':bitcoinCommentsBlob, 'ethereum_comments':ethereumCommentsBlob, 'monero_comments':moneroCommentsBlob, 'dashpay_comments':dashpayCommentsBlob, 'ethtrader_comments':ethtraderCommentsBlob, 'ethfinance_comments':ethfinanceCommentsBlob, 'xmrtrader_comments':xmrtraderCommentsBlob})

# Clean up the comments

def _removeNonAscii(s):
    return "".join(i for i in s if  ord(i)<128)

def make_lower_case(text):
    return text.lower()

def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text

def clean(x):
    x = re.sub(r'^RT[\s]+', '', x)
    x = re.sub(r'https?:\/\/.*[\r\n]*', '', x)
    x = re.sub(r'#', '', x)
    x = re.sub(r'@[A-Za-z0–9]+', '', x) 
    return x

def lemmatize_stemming(text):
    return ''.join(lemtzer.lemmatize(text, pos='v'))

def preprocess(text):
    result=[]
    for token in simple_preprocess(text):
        if token not in STOPWORDS:
            result.append(lemmatize_stemming(token))
   
    return ' '.join(result)


def cleanComments(coinColName):
    dfComments[coinColName] = dfComments[coinColName].astype(str).apply(_removeNonAscii)
    dfComments[coinColName] = dfComments[coinColName].apply(func = make_lower_case)
    dfComments[coinColName] = dfComments[coinColName].apply(func=remove_punctuation)
    dfComments[coinColName] = dfComments[coinColName].apply(func=remove_html)
    dfComments[coinColName] = dfComments[coinColName].apply(func=clean)
    dfComments[coinColName] = dfComments[coinColName].apply(func=lemmatize_stemming)
    dfComments[coinColName] = dfComments[coinColName].apply(func=preprocess)


stop_words = set(stopwords.words('english'))  
lemtzer = WordNetLemmatizer()


colList = ["btc_comments", "bitcoin_comments", "ethereum_comments", "monero_comments", "dashpay_comments", "ethtrader_comments", "ethfinance_comments", "xmrtrader_comments"]

for coin in colList:
    cleanComments(coin)
    
    
#Function for polarity and subjectivity
polarity = lambda x: TextBlob(x).sentiment.polarity
subjectivity = lambda x: TextBlob(x).sentiment.subjectivity

#Passing the function to calculate polarity and subjectivity
dfComments['btc_polarity'] = dfComments['btc_comments'].apply(polarity)
dfComments['btc_subjectivity'] = dfComments['btc_comments'].apply(subjectivity)
dfComments['bitcoin_polarity'] = dfComments['bitcoin_comments'].apply(polarity)
dfComments['bitcoin_subjectivity'] = dfComments['bitcoin_comments'].apply(subjectivity)
dfComments['ethereum_polarity'] = dfComments['ethereum_comments'].apply(polarity)
dfComments['ethereum_subjectivity'] = dfComments['ethereum_comments'].apply(subjectivity)
dfComments['monero_polarity'] = dfComments['monero_comments'].apply(polarity)
dfComments['monero_subjectivity'] = dfComments['monero_comments'].apply(subjectivity)
dfComments['dashpay_polarity'] = dfComments['dashpay_comments'].apply(polarity)
dfComments['dashpay_subjectivity'] = dfComments['dashpay_comments'].apply(subjectivity)
dfComments['ethtrader_polarity'] = dfComments['ethtrader_comments'].apply(polarity)
dfComments['ethtrader_subjectivity'] = dfComments['ethtrader_comments'].apply(subjectivity)
dfComments['ethfinance_polarity'] = dfComments['ethfinance_comments'].apply(polarity)
dfComments['ethfinance_subjectivity'] = dfComments['ethfinance_comments'].apply(subjectivity)
dfComments['xmrtrader_polarity'] = dfComments['xmrtrader_comments'].apply(polarity)
dfComments['xmrtrader_subjectivity'] = dfComments['xmrtrader_comments'].apply(subjectivity)

# Rename columns
df['btc_polarity'] = dfComments['btc_polarity']
df['btc_subjectivity'] = dfComments['btc_subjectivity']
df['bitcoin_polarity'] = dfComments['bitcoin_polarity']
df['bitcoin_subjectivity'] = dfComments['bitcoin_subjectivity'] 
df['ethereum_polarity'] = dfComments['ethereum_polarity']
df['ethereum_subjectivity'] =dfComments['ethereum_subjectivity'] 
df['monero_polarity'] = dfComments['monero_polarity']
df['monero_subjectivity'] =dfComments['monero_subjectivity'] 
df['dashpay_polarity'] = dfComments['dashpay_polarity']
df['dashpay_subjectivity'] = dfComments['dashpay_subjectivity'] 
df['ethtrader_polarity'] = dfComments['ethtrader_polarity'] 
df['ethtrader_subjectivity'] = dfComments['ethtrader_subjectivity'] 
df['ethfinance_polarity'] = dfComments['ethfinance_polarity'] 
df['ethfinance_subjectivity'] = dfComments['ethfinance_subjectivity'] 
df['xmrtrader_polarity'] = dfComments['xmrtrader_polarity'] 
df['xmrtrader_subjectivity'] = dfComments['xmrtrader_subjectivity']

df['btc_comments'] = dfComments['btc_comments']
df['bitcoin_comments'] = dfComments['bitcoin_comments']
df['ethereum_comments'] = dfComments['ethereum_comments']
df['ethtrader_comments'] = dfComments['ethtrader_comments']
df['ethfinance_comments'] = dfComments['ethfinance_comments']
df['dashpay_comments'] = dfComments['dashpay_comments']
df['monero_comments'] = dfComments['monero_comments']
df['xmrtrader_comments'] = dfComments['xmrtrader_comments']

df["btc_symbol"] = "BTC"
df["ethereum_symbol"] = "ETH"
df["ethfinance_symbol"] = "ETH"
df["ethtrader_symbol"] = "ETH"
df["bitcoin_symbol"] = "BITCOIN"
df["dashpay_symbol"] = "DASH"
df["monero_symbol"] = "XMR"
df["xmrtrader_symbol"] = "XMR"

df["ethtrader_volume"] = df["ethereum_volume"]
df["ethfinance_volume"] = df["ethereum_volume"]
df["xmrtrader_volume"] = df["monero_volume"]


#Clean (tidy) the data
 
tidyData = pd.DataFrame(columns = ["hour", "online_users", "number_of_post", "comments", "total_votes", "polarity", "subjectivity", "price", "signal", "pctChange", "volume", 'symbol'])
subRedditName = ['bitcoin', 'btc', 'ethereum', 'monero', 'dashpay', 'ethtrader', 'ethfinance', 'xmrtrader' ] 

for subName in subRedditName:
    sufVals = ["online_users", "number_of_post", "comments", "total_votes", "polarity", "subjectivity", "price", "signal", "pctChange", "volume", 'symbol']
    chunk = df[["Hour"] + [subName + "_{}".format(i) for i in sufVals]]
    chunk.columns = ['hour'] + sufVals
    chunk['subReddit'] = subName
    tidyData = pd.concat([tidyData, chunk])
    

tidyData = pd.DataFrame(tidyData)


# More NLP to do 2nd sentiment analysis and make doc2vec (did not work very well, dont use)

newComments = tidyData

sid = SentimentIntensityAnalyzer()
newComments["sentiments"] = tidyData["comments"].apply(lambda x: sid.polarity_scores(x))
newComments = pd.concat([newComments.drop(['sentiments'], axis=1), newComments['sentiments'].apply(pd.Series)], axis=1)

tidyData = newComments
newNLP = tidyData.comments

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(newNLP)]
# try differnt vector sizes!
model = Doc2Vec(documents, vector_size=50, window=5, min_count=1, workers=4)
newNLP2 = pd.DataFrame([model.docvecs[i] for i in range(len(newNLP))])


# Clean data more, number of active user numbners (remove "k"), fix data time format, remove nan
def removeKs(dfNum):
    dfNum = str(dfNum).lstrip('0')
    dfNum = str(dfNum).lstrip()
    if len(dfNum) == 0:
        return 0
    
    if dfNum[-1] == 'k':
        return int(float(dfNum[0:-1]) * 1000)
    
    return int(float(dfNum))

tidyData["online_users"] = tidyData["online_users"].map(lambda x: removeKs(x))


# Date/time format and generate day of week and hour of day feature
tidyData = tidyData.reset_index(drop=True)
tidyData['hour'] = pd.to_datetime(tidyData['hour'])

tidyData['hour'] = pd.to_datetime(tidyData['hour'], format = '%m/%d/%y %H:%M %p')

tidyData['day_of_week'] = tidyData.hour.dt.dayofweek
tidyData['hour_of_day'] = tidyData.hour.dt.hour


# Remove Na's
tidyData['number_of_post'] = tidyData['number_of_post'].fillna(0)
tidyData['total_votes'] = tidyData['total_votes'].fillna(0)
tidyData['pctChange'] = tidyData['pctChange'].fillna(0)

# Generate list of most important words

commentsSignal = tidyData.comments
count_vec = CountVectorizer()
X_train_count = count_vec.fit_transform(commentsSignal)
X_train_count.shape
X_train_count

clf = DecisionTreeClassifier(random_state=0)
commnetModel = clf.fit(X_train_count, tidyData.signal.astype(int))

### Make a list of important words and score them then count them 
vectWords = pd.DataFrame(count_vec.get_feature_names())
wordScores = pd.DataFrame(commnetModel.feature_importances_)

vectWords.columns=['word']
wordScores.columns=['score']

wordsAndScores = wordScores.join(vectWords)

dropRows = wordsAndScores[wordsAndScores['score'] == 0].index


# Delete these row indexes from dataFrame and make a word list
wordsAndScores.drop(dropRows , inplace=True)

wordScores = wordsAndScores.sort_values('score', ascending = False).iloc[:10]

wordsOnly = wordScores.drop(columns='score')

wordsOnly.reset_index(drop=True, inplace=True)
wordsOnlyT = wordsOnly.T
wordList = wordsOnly.word.tolist()

headRow=1
wordsOnlyT.columns = wordsOnlyT.iloc[0]
wordsOnlyT.reset_index(drop=True, inplace=True)

wideWithWords = pd.concat([tidyData.comments, wordsOnlyT])
wideWithWords.rename(columns={ wideWithWords.columns[0]: "comments" }, inplace = True)

# Make the counts and set new features!!! (Only important part)
g=pd.get_dummies(pd.Series(wideWithWords.comments.str.split('\s').explode())).reindex(columns=wordList).fillna(0).astype(int)
wordCounts = pd.DataFrame(wideWithWords.iloc[:,0]).join(g.groupby(level=0).sum(0)).drop(columns='comments')

wordCounts = wordCounts[:-1]
wordCounts = wordCounts.reset_index(drop=True)

# Add words to tidydata
tidyData = pd.concat([tidyData, wordCounts], axis = 1)

# Make output for important words
outputFive = wordCounts.sum()
outputFive = outputFive.sort_values(ascending=False)


# In[291]:


##################################
#####   SET FEATURES HERE   ######


# HERE IS WHERE WE INPUT THE USER'S SELECTED FEATURES

# FEATURES to choose from:
#  ['hour_of_day', 'day_of_week', 'online_users', 'number_of_post', 'total_votes', 'polarity',
#   'subjectivity', 'volume', 'negativity', 'neutrality', 'positivty', 'top_10_words']




features = tidyData.copy().drop(columns =["comments", 'pctChange', 'price', 'volume'], axis=1)


dummyFeatures = pd.get_dummies(features)


# In[ ]:


# Get Dummies

### Test dummify day of week
# dummyDays = pd.get_dummies(dummyFeatures['day_of_week'])
# dummyFeatures = dummyFeatures.drop(columns =["day_of_week"], axis=1)
# dummyFeatures = pd.concat([dummyFeatures, dummyDays], axis = 1)


dummyHours = pd.get_dummies(dummyFeatures['hour_of_day'])
dummyFeatures = dummyFeatures.drop(columns =["hour_of_day"], axis=1)
dummyFeatures = pd.concat([dummyFeatures, dummyHours], axis = 1)


dummyFeatures = dummyFeatures.sort_values('hour')
dummyFeatures.columns


target = dummyFeatures.signal_1.astype(int)

dummyFeatures = dummyFeatures.drop(columns =['hour','signal_1','signal_0', 'symbol_BITCOIN', 'symbol_BTC', 'symbol_DASH', 'symbol_ETH',
       'symbol_XMR'], axis=1)


# Split the NEW data that is to be predicted

newRowToPredict = dummyFeatures.iloc[-8:]
dummyFeatures = dummyFeatures.iloc[:-8]


targetToPredict = target.iloc[-8:]
target = target.iloc[:-8]


# Train test split 
featuresTrain, featuresTest, signalTrain, signalTest = train_test_split(dummyFeatures, target, test_size=0.2, random_state=42)


# Run the RF model
rf_model = RandomForestClassifier()
param_grid = {'max_depth':np.arange(5, 40, 5), 'min_samples_split':np.arange(2, 10, 2)}
gsModelTrain = GridSearchCV(estimator = rf_model, param_grid = param_grid, cv=3, scoring = 'roc_auc')
    

gsModelTrain.fit(featuresTrain, signalTrain)

rf_model.set_params(**gsModelTrain.best_params_)

rf_model.fit(featuresTrain, signalTrain)

testResults = rf_model.predict(featuresTest)


# In[ ]:


###############
### OUTPUTS ###


target_names = ['class 0', 'class 1']
classReport = classification_report(signalTest, testResults, target_names=target_names, output_dict=True)



outputOne = pd.DataFrame(classReport).transpose()

outputTwo = f'\nr^2 Score: {rf_model.score(featuresTest, signalTest)}'

outputSignalXMR = (f'\nSignal prediction: \nMonero:    {rf_model.predict(newRowToPredict.values.reshape(8,-1))[0]}')
outputSignalEthTrader = (f'EthTrader: {rf_model.predict(newRowToPredict.values.reshape(8,-1))[1]}')
outputSignalEthDash = (f'Dash:      {rf_model.predict(newRowToPredict.values.reshape(8,-1))[2]}')
outputSignalEthereum = (f'Ethereum:  {rf_model.predict(newRowToPredict.values.reshape(8,-1))[3]}')
outputSignalBTC = (f'BCH:       {rf_model.predict(newRowToPredict.values.reshape(8,-1))[4]}')
outputSignalXMRTrader = (f'XMRTrader: {rf_model.predict(newRowToPredict.values.reshape(8,-1))[7]}')
outputSignalBitcoin = (f'Bitcoin:   {rf_model.predict(newRowToPredict.values.reshape(8,-1))[5]}')
outputSignalEthFinance = (f'EthFinance:{rf_model.predict(newRowToPredict.values.reshape(8,-1))[6]}')
                         
outputFour = f'\nFeature Importances: \n{pd.Series(rf_model.feature_importances_, index = dummyFeatures.columns).head(10)}'

print(outputOne)
print(outputTwo)
print(outputSignalXMR)
print(outputSignalEthTrader)
print(outputSignalEthDash)
print(outputSignalEthereum)
print(outputSignalBTC)
print(outputSignalXMRTrader)
print(outputSignalBitcoin)
print(outputSignalEthFinance)
print(outputFour)

print(f'\nImportant words: \n{outputFive}')



# print signals to CSV

allSignals = rf_model.predict(newRowToPredict.values.reshape(8,-1))
signalInput = pd.read_csv ('SignalInput.csv')

if allSignals[6] == 0:
    btcSignal = 0
else:
    btcSignal =1

    
if allSignals[0] and allSignals[5] == 1:
    xmrSignal = 1
else:
    xmrSignal =0
    
    
if allSignals[1] + allSignals[3] + allSignals[7] > 1:
    ethSignal = 1
else:
    ethSignal = 0
    
    
if allSignals[2] == 0:
    dashSignal = 0
else:
    dashSignal = 1
    
    
if allSignals[4] == 0:
    bchSignal = 0
else:
    bchSignal = 1

newSignals = pd.DataFrame(np.array([btcSignal, bchSignal, ethSignal, xmrSignal, dashSignal]).reshape(1,5), columns=signalInput.columns)


newSignalInput = pd.concat([signalInput, newSignals])
newSignalInput.to_csv('SignalInput.csv', index=False)



# Model split by SubReddit

dummySubRedditFeatures = features.copy()
dummyHours = pd.get_dummies(dummySubRedditFeatures['hour_of_day'])
dummySubRedditFeatures = pd.concat([dummySubRedditFeatures, dummyHours], axis = 1)


### Test day of week
# dummyDays = pd.get_dummies(dummyFeatures['day_of_week'])
# dummyFeatures = dummyFeatures.drop(columns =["day_of_week"], axis=1)
# dummyFeatures = pd.concat([dummyFeatures, dummyDays], axis = 1)


dummySubRedditFeatures = dummySubRedditFeatures.sort_values('hour')
dummySubRedditFeatures.signal = dummySubRedditFeatures.signal.astype(int)

# Split NEW data out to predict
subRedditRowsToPredict = dummySubRedditFeatures.iloc[-8:]
dummySubRedditFeatures = dummySubRedditFeatures.iloc[:-8]


# Split by subreddit and train test split

subRedditDict = {}
subRedditFeaturesTrain ={}
subRedditFeaturesTest ={}
subRedditSignalTrain ={}
subRedditSignalTest ={}

for subReddit, subRedditDf in dummySubRedditFeatures.groupby('subReddit'):
    
    subRedditFeaturesTrain[subReddit], subRedditFeaturesTest[subReddit], subRedditSignalTrain[subReddit], subRedditSignalTest[subReddit] = train_test_split(dummySubRedditFeatures.drop(['subReddit','signal','hour','symbol'], axis =1), dummySubRedditFeatures.signal, test_size=0.2, random_state=42)

    rf_model = RandomForestRegressor()
    param_grid = {'max_depth':np.arange(5, 10, 5), 'min_samples_split':np.arange(2, 6, 2)}
    gsModelTrain = GridSearchCV(estimator = rf_model, param_grid = param_grid, cv=2)


    gsModelTrain.fit(subRedditFeaturesTrain[subReddit], subRedditSignalTrain[subReddit])

    rf_model.set_params(**gsModelTrain.best_params_)


    # Fit to train data
    rf_model.fit(subRedditFeaturesTrain[subReddit], subRedditSignalTrain[subReddit])
    
    ###

    #Get scores comparing real signals and predicted signals from the test dataset.
    print("r2 Train score:", subReddit, r2_score(subRedditSignalTrain[subReddit], rf_model.predict(subRedditFeaturesTrain[subReddit])))
    print("r2 Test score:", subReddit, r2_score(subRedditSignalTest[subReddit], rf_model.predict(subRedditFeaturesTest[subReddit])))
    trainRMSE = np.sqrt(mean_squared_error(y_true=subRedditSignalTrain[subReddit], y_pred=rf_model.predict(subRedditFeaturesTrain[subReddit])))
    testRMSE = np.sqrt(mean_squared_error(y_true=subRedditSignalTest[subReddit], y_pred=rf_model.predict(subRedditFeaturesTest[subReddit])))
    print("Train RMSE:", subReddit, trainRMSE)
    print("Test RMSE:", subReddit, testRMSE)
    


# In[ ]:


dummySubRedditFeatures = dummySubRedditFeatures.drop(['hour','symbol', 'signal', 'subReddit'], axis=1)


# In[ ]:


subRedditRowsToPredict = subRedditRowsToPredict.drop(columns=['subReddit','signal','hour','symbol'])

# 


# In[ ]:


print(rf_model.predict(subRedditRowsToPredict.values.reshape(8,-1)))


# In[ ]:



sys.stdout = old_stdout

log_file.close()
