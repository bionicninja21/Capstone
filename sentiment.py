import time


from keras.preprocessing.sequence import pad_sequences
import pickle

start=time.time()

def encode_text(tokenizer, lines, length):
    # integer encode
    encoded = tokenizer.texts_to_sequences(lines)
    # pad encoded sequences
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded

with open('C:/Users/bioni/Documents/Capstone/Saved_models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)  
length =800    
from tensorflow import keras
model = keras.models.load_model('C:/Users/bioni/Documents/Capstone/senti.h5')

estimate=model.predict(encode_text(tokenizer,["This product is a dissapointment. Please dont waste money to buy it."],length))
rate=estimate[0]*50
print (rate[0])
end=time.time()
print("Execution time:",end-start)