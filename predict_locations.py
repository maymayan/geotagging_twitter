import csv
import pickle
import joblib
import jpype as jp
import numpy as np
import pandas as pd
import sys


input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

ZEMBEREK_PATH = 'zemberek/zemberek-full.jar'

jp.startJVM(jp.getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % (ZEMBEREK_PATH))

TurkishTokenizer = jp.JClass('zemberek.tokenization.TurkishTokenizer')
TurkishLexer = jp.JClass('zemberek.tokenization.antlr.TurkishLexer')

tokenizer = TurkishTokenizer.DEFAULT

TurkishMorphology = jp.JClass('zemberek.morphology.TurkishMorphology')
morphology = TurkishMorphology.createWithDefaults()

# stopwords extraction from text file
stop_words = []
with open('data/stop-words.tr.txt', 'r', encoding="utf-8") as stopFile:
    reader = csv.reader(stopFile)
    for stop_word in reader:
        stop_words.append(stop_word)

# read the input_file to DataFrame using pandas
df = pd.read_csv(input_file_path)

# stopwords, lemmatization, stemming and tokenization on tweets of users
stopped_tweets = []
for tweet in df['tweet']:
    stop_words_tweet = ""
    tokens = tokenizer.tokenize(tweet)
    for token in tokens:
        token_text = token.getText()
        if token_text[:1] == '#':
            token_text = token_text[1:]
        for i in range(len(stop_words)):
            s_word = stop_words[i][0]
            if token_text == s_word or token_text == ',':
                break
            if i == len(stop_words) - 1:
                analysis = morphology.analyzeSentence(token_text)

                results = morphology.disambiguate(token_text, analysis).bestAnalysis()
                lemma = results[0].getLemmas()[0]
                if not 'UNK' == lemma:
                    stop_words_tweet = stop_words_tweet + lemma + " "
    stopped_tweets.append(stop_words_tweet)

# stopwords, lemmatization, stemming and tokenization on bio informations of users
stopped_bios = []
for bio in df['bio']:
    stop_words_bio = ""
    tokens = tokenizer.tokenize(bio)
    for token in tokens:
        token_text = token.getText()
        if token_text[:1] == '#':
            token_text = token_text[1:]
        for i in range(len(stop_words)):
            s_word = stop_words[i][0]
            if token_text == s_word or token_text == ',':
                break
            if i == len(stop_words) - 1:
                analysis = morphology.analyzeSentence(token_text)

                results = morphology.disambiguate(token_text, analysis).bestAnalysis()
                lemma = results[0].getLemmas()[0]
                if not 'UNK' == lemma:
                    stop_words_bio = stop_words_bio + lemma + " "
    stopped_bios.append(stop_words_bio)

# swap tweets and bios by stoppeds
df['tweet'] = stopped_tweets
df['bio'] = stopped_bios

TurkishMorphology = jp.JClass('zemberek.morphology.TurkishMorphology')
Paths = jp.JClass('java.nio.file.Paths')

morphology = TurkishMorphology.createWithDefaults()

modelPath = Paths.get('ner-model')

PerceptronNer = jp.JClass('zemberek.ner.PerceptronNer')

ner = PerceptronNer.loadModel(modelPath, morphology)


def get_named_entities(input_str):
    result = ner.findNamedEntities(input_str)
    named_entities_list = result.getNamedEntities()

    named_entities_str = ""
    for item in named_entities_list:
        named_entities_str += item.toString() + " "
    return named_entities_str


# feature extraction by using named entity recognition
bio_entities = []
for e in df['bio']:
    bio_entities.append(get_named_entities(e))
df['bio_entities'] = bio_entities
tweeet_entities = []
for e in df['tweet']:
    tweeet_entities.append(get_named_entities(e))
df['tweet_entities'] = tweeet_entities

# for all text-contented attributes, create tf-idf vectors by using vectorizer models we saved before
data_texts = df[['bio', 'tweet', 'hashtag', 'bio_entities', 'tweet_entities']]
for col in data_texts.columns:
    vectorizer = pickle.load(open("vectorizer-models/" + col + ".vectorizer.pickle", "rb"))

    x = vectorizer.transform(df[col].values.astype('U'))
    df1 = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names())
    df = pd.concat([df, df1], axis=1)

# dropping useless features
df = df.drop('tweet', axis=1)
df = df.drop('bio', axis=1)
df = df.drop('hashtag', axis=1)
df = df.drop('bio_entities', axis=1)
df = df.drop('tweet_entities', axis=1)
df = df.drop('nickname', axis=1)

# representing the data as an np array to use by model that we saved.
X = np.array(df)

# loading the model
filename = 'classification-model/geotag_model.sav'
loaded_model = joblib.load(filename)

# the predictions on given input data are being written to output_file
predictions = loaded_model.predict(X)
with open(output_file_path, 'a') as output_file:
    for prediction in predictions:
        print(prediction, file=output_file)
