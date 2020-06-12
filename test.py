from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.externals import joblib

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
source = input('Put an answer\n')
text = [source]
text = tokenizer.texts_to_sequences(text)
text = pad_sequences(text, maxlen=28, dtype='int32', value=0)
model = joblib.load("./models/sentiment_model.pkl")
sentiment = model.predict(text,batch_size=1,verbose = 3)[0]
print(sentiment)
