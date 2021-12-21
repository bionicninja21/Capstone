import sys
sys.path.append('C:/Users/bioni/Documents/Capstone')
import time
import pickle
import keras
from keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
nltk.download('punkt')
from attention import AttentionLayer
import numpy as np
import pandas as pd 
import warnings
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")

start=time.time()
review="always perfect snack dog loves knows exactly starts ask time evening gets greenie snack thank excellent product fast delivery  "


def f1(review):
    with open('C:/Users/bioni/Documents/Capstone/filter.pkl', 'rb') as f:
        classifier = pickle.load(f)
        
    table = str.maketrans({key: None for key in string.punctuation})
    def preProcess(text):
        lemmatizer = WordNetLemmatizer()
        filtered_tokens=[]
        lemmatized_tokens = []
        stop_words = set(stopwords.words('english'))
        text = text.translate(table)
        for w in text.split(" "):
            if w not in stop_words:
                lemmatized_tokens.append(lemmatizer.lemmatize(w.lower()))
                filtered_tokens = [' '.join(l) for l in nltk.bigrams(lemmatized_tokens)] + lemmatized_tokens
                return filtered_tokens

    featureDict = {}
    def toFeatureVector(tokens):
        localDict = {}
        for token in tokens:
            if token not in featureDict:
                featureDict[token] = 1
            else:
                featureDict[token] = +1
   
            if token not in localDict:
                localDict[token] = 1
            else:
                localDict[token] = +1
    
        return localDict
    
    def predictLabel(reviewSample, classifier):
        return classifier.classify(toFeatureVector(preProcess(reviewSample)))
    
    op=predictLabel(review,classifier)
    
    if op=="real":
        print("Not Suspicious")
    else:
        print ("Suspicious")

def f2(review):
    max_text_len=30
    max_summary_len=8
    
    with open('C:/Users/bioni/Documents/Capstone/summary_tockenizer.pkl', 'rb') as f:
        x_tokenizer = pickle.load(f)
    
    with open('C:/Users/bioni/Documents/Capstone/target_word_index.pkl', 'rb') as f:
        target_word_index = pickle.load(f)
    
    with open('C:/Users/bioni/Documents/Capstone/reverse_target_word_index.pkl', 'rb') as f:
        reverse_target_word_index = pickle.load(f)
   # with open('C:/Users/bioni/Documents/Capstone/embeddings.pkl', 'rb') as f:
        #embeddings_index = pickle.load(f)   
    
    encoder_model=keras.models.load_model('C:/Users/bioni/Documents/Capstone/encoder_model.h5')
    decoder_model=keras.models.load_model('C:/Users/bioni/Documents/Capstone/decoder_model.h5',custom_objects={'AttentionLayer': AttentionLayer})
    
    def predict_summary(input):
      input_list =[input]
      input_array = np.array(input_list)
      input_seq = x_tokenizer.texts_to_sequences(input_array)
      input_seq = np.array(input_seq[0],dtype=np.int32)
      tokenized_list = list()
      tokenized_list.append(input_seq)
      padded_input = pad_sequences(tokenized_list,  maxlen=max_text_len, padding='post')
      return decode_sequence(padded_input[0].reshape(1,max_text_len))
    
    def decode_sequence(input_seq):
        e_out, e_h, e_c = encoder_model.predict(input_seq)
        target_seq = np.zeros((1,1))
    
        target_seq[0, 0] = target_word_index['sostok']

        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
          
            output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
    
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = reverse_target_word_index[sampled_token_index]
            
            if(sampled_token!='eostok'):
                decoded_sentence += ' '+sampled_token
    
            if (sampled_token == 'eostok'  or len(decoded_sentence.split()) >= (max_summary_len-1)):
                stop_condition = True

            target_seq = np.zeros((1,1))
            target_seq[0, 0] = sampled_token_index
    
            e_h, e_c = h, c
    
        return decoded_sentence
    
    print(predict_summary(review))

def f3(review):
    length =800    
    model = keras.models.load_model('C:/Users/bioni/Documents/Capstone/senti.h5')
    
    with open('C:/Users/bioni/Documents/Capstone/Saved_models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)  
    def encode_text(tokenizer, lines, length):
        encoded = tokenizer.texts_to_sequences(lines)
        padded = pad_sequences(encoded, maxlen=length, padding='post')
        return padded
    print(model.predict(encode_text(tokenizer,[review],length)))



f1(review)
f2(review)       
f3(review)     

end=time.time()
print("Execution time:",end-start)



