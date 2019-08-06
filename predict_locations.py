import numpy as np
import pandas as pd
import joblib
import pickle

df = pd.read_csv('data/test.csv')

data_texts = df[['bio', 'tweet', 'hashtag', 'bio_entities', 'tweet_entities']]
for col in data_texts.columns:
    vectorizer = pickle.load(open("vectorizer-models/"+col+".vectorizer.pickle", "rb"))

    x = vectorizer.transform(df[col].values.astype('U'))
    df1 = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names())
    df = pd.concat([df, df1], axis=1)

df = df.drop('tweet', axis=1)
df = df.drop('bio', axis=1)
df = df.drop('hashtag', axis=1)
df = df.drop('bio_entities', axis=1)
df = df.drop('tweet_entities', axis=1)
df = df.drop('gps', axis=1)
df = df.drop('nickname', axis=1)
y = df.iloc[:, 0].values
df = df.drop('geotag', axis=1)
X = np.array(df)

filename = 'classification-model/geotag_model.sav'
loaded_model = joblib.load(filename)
predictions = loaded_model.predict(X)

print(predictions)
