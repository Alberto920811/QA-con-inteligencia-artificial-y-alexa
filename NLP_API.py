from flask import Flask,jsonify,request

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.externals import joblib

app = Flask(__name__)

@app.route("/", methods=['POST','GET'])
def index():
    if(request.method == 'POST'):
        max_fatures = 2000

        model = joblib.load("./models/sentiment_model.pkl")
        data = request.get_json()
        tokenizer = Tokenizer(num_words=max_fatures, split=' ')
        twt = tokenizer.texts_to_sequences(text)
        twt = pad_sequences(twt, maxlen=28, dtype='int32', value=0)
        
        return jsonify(model.predict(twt,batch_size=1,verbose = 3)[0].tolist())
    else:
        return  jsonify({"about":"Error"})

if __name__ == '__main__':
    app.run(debug=True)