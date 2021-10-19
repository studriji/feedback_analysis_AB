from flask import Flask, app,render_template,url_for,request
import pandas as pd 
import numpy as np
from tensorflow import keras
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

model = keras.models.load_model("model.h5")
model.load_weights('feedback_model.h5')
tokenizer = pickle.load(open('tokenizer.pkl','rb'))

app=Flask(__name__)
@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		feedback = request.form['message']
		feedback = [feedback]
		#print(feedback)
		feedback = tokenizer.texts_to_sequences(feedback)
		#print(feedback)
		#padding the tweet to have exactly the same shape 
		feedback = pad_sequences(feedback, maxlen=28, dtype='int32', value=0)
		#print(feedback)
		sentiment = model.predict(feedback,batch_size=1,verbose = 'auto')[0]
		#print(sentiment)
		#print(np.argmax(sentiment))
		if(np.argmax(sentiment) == 0):
			print("negative")
			#result = 'This is a negative feedback'
			return render_template('result.html',prediction=0)
		elif(np.argmax(sentiment) == 1):
			print("positive")
			#result = 'This is a positive feedback'
			return render_template('result.html',prediction=1)
    	 	 
			 

if __name__ == '__main__':
	app.run(debug=True)


'''
model = keras.models.load_model("model.h5")
#model.load_weights('feedback_model.h5')
print("model called")
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
max_fatures = 200
tokenizer = Tokenizer(num_words=max_fatures,split=' ')
print("tokenizer declaring",tokenizer)
feedback = ['the food of the resturant is very good']
#vectorizing the feedback by the pre-fitted tokenizer instance
tokenizer = pickle.load(open('tokenizer.pkl','rb'))
feedback = tokenizer.texts_to_sequences(feedback)
print(feedback)
#padding the tweet to have exactly the same shape 
feedback = pad_sequences(feedback, maxlen=28, dtype='int32', value=0)
print(feedback)
sentiment = model.predict(feedback,batch_size=1,verbose = 'auto')[0]
print(sentiment)
print(np.argmax(sentiment))
if(np.argmax(sentiment) == 0):
    print("negative")
elif (np.argmax(sentiment) == 1):
    print("positive")
	'''