import sys

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

import project_support

from tkinter import messagebox
from multiprocessing import Process, Queue,Pool
import threading
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


def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = tk.Tk()
    top = Toplevel1 (root)
    project_support.init(root, top)
    root.mainloop()

w = None
def create_Toplevel1(rt, *args, **kwargs):
    '''Starting point when module is imported by another module.
       Correct form of call: 'create_Toplevel1(root, *args, **kwargs)' .'''
    global w, w_win, root
    #rt = root
    root = rt
    w = tk.Toplevel (root)
    top = Toplevel1 (w)
    project_support.init(w, top, *args, **kwargs)
    return (w, top)

def destroy_Toplevel1():
    global w
    w.destroy()
    w = None


class Toplevel1:
    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'
        _ana1color = '#d9d9d9' # X11 color: 'gray85'
        _ana2color = '#ececec' # Closest X11 color: 'gray92'

        top.geometry("600x450+660+210")
        top.minsize(148, 1)
        top.maxsize(1924, 1055)
        top.resizable(1,  1)
        top.title("Capstone Project")
        top.configure(background="#d9d9d9")

        self.Entry1 = tk.Entry(top)
        self.Entry1.place(relx=0.3, rely=0.289, height=84, relwidth=0.59)
        self.Entry1.configure(background="white")
        self.Entry1.configure(cursor="fleur")
        self.Entry1.configure(disabledforeground="#a3a3a3")
        self.Entry1.configure(font="TkFixedFont")
        self.Entry1.configure(foreground="#000000")
        self.Entry1.configure(insertbackground="black")

        def g1():
            review=self.Entry1.get()
    
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
            
            if op!="real":
                messagebox.showinfo("Review Classification","         Not Suspicious        ")
            else:
                messagebox.showinfo("Review Classification","         Suspicious        ")
        
        

            
        self.Label1 = tk.Label(top)
        self.Label1.place(relx=0.017, rely=0.311, height=66, width=162)
        self.Label1.configure(background="#d9d9d9")
        self.Label1.configure(cursor="fleur")
        self.Label1.configure(disabledforeground="#a3a3a3")
        self.Label1.configure(font="-family {Georgia} -size 16")
        self.Label1.configure(foreground="#000000")
        self.Label1.configure(text='''Review''')

        self.Button1 = tk.Button(top)
        self.Button1.place(relx=0.35, rely=0.556, height=45, width=205)
        self.Button1.configure(activebackground="#ececec")
        self.Button1.configure(activeforeground="#000000")
        self.Button1.configure(background="#d9d9d9")
        self.Button1.configure(cursor="fleur")
        self.Button1.configure(disabledforeground="#a3a3a3")
        self.Button1.configure(font="-family {Segoe UI} -size 9")
        self.Button1.configure(foreground="#000000")
        self.Button1.configure(highlightbackground="#d9d9d9")
        self.Button1.configure(highlightcolor="black")
        self.Button1.configure(pady="0")
        self.Button1.configure(text='''Suspicious Review Classifier''')
        self.Button1.configure(command=g1)
        
        def g2():
            review=self.Entry1.get()
            max_text_len=30
            max_summary_len=8
            
            with open('C:/Users/bioni/Documents/Capstone/summary_tockenizer.pkl', 'rb') as f:
                x_tokenizer = pickle.load(f)
            
            with open('C:/Users/bioni/Documents/Capstone/target_word_index.pkl', 'rb') as f:
                target_word_index = pickle.load(f)
            
            with open('C:/Users/bioni/Documents/Capstone/reverse_target_word_index.pkl', 'rb') as f:
                reverse_target_word_index = pickle.load(f)
            
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
            
            summary=predict_summary(review)
            messagebox.showinfo("Review Summary","        " +summary+ "        ")
            

        self.Button2 = tk.Button(top)
        self.Button2.place(relx=0.05, rely=0.556, height=45, width=160)
        self.Button2.configure(activebackground="#ececec")
        self.Button2.configure(activeforeground="#000000")
        self.Button2.configure(background="#d9d9d9")
        self.Button2.configure(disabledforeground="#a3a3a3")
        self.Button2.configure(foreground="#000000")
        self.Button2.configure(highlightbackground="#d9d9d9")
        self.Button2.configure(highlightcolor="black")
        self.Button2.configure(pady="0")
        self.Button2.configure(text='''Review Summarizer''')
        self.Button2.configure(command=g2)

        def g3():
            review=self.Entry1.get()
            
            length =800    
            model = keras.models.load_model('C:/Users/bioni/Documents/Capstone/senti.h5')
            
            with open('C:/Users/bioni/Documents/Capstone/Saved_models/tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)  
            def encode_text(tokenizer, lines, length):
                encoded = tokenizer.texts_to_sequences(lines)
                padded = pad_sequences(encoded, maxlen=length, padding='post')
                return padded
            
            estimate=model.predict(encode_text(tokenizer,[review],length))[0][0]*5
            rating= round(estimate,1)
            messagebox.showinfo("Review Rating","        " +str(rating)+ "/5        ")

        self.Button3 = tk.Button(top)
        self.Button3.place(relx=0.733, rely=0.556, height=45, width=140)
        self.Button3.configure(activebackground="#ececec")
        self.Button3.configure(activeforeground="#000000")
        self.Button3.configure(background="#d9d9d9")
        self.Button3.configure(disabledforeground="#a3a3a3")
        self.Button3.configure(foreground="#000000")
        self.Button3.configure(highlightbackground="#d9d9d9")
        self.Button3.configure(highlightcolor="black")
        self.Button3.configure(pady="0")
        self.Button3.configure(text='''Rating Predictor''')
        self.Button3.configure(command=g3)

        
        def g4():
            review=self.Entry1.get()
            
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
            
            max_text_len=30
            max_summary_len=8
            
            with open('C:/Users/bioni/Documents/Capstone/summary_tockenizer.pkl', 'rb') as f:
                x_tokenizer = pickle.load(f)
            
            with open('C:/Users/bioni/Documents/Capstone/target_word_index.pkl', 'rb') as f:
                target_word_index = pickle.load(f)
            
            with open('C:/Users/bioni/Documents/Capstone/reverse_target_word_index.pkl', 'rb') as f:
                reverse_target_word_index = pickle.load(f)
            
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
            
            summary=predict_summary(review)
            
            length =800    
            model1 = keras.models.load_model('C:/Users/bioni/Documents/Capstone/senti.h5')
            
            with open('C:/Users/bioni/Documents/Capstone/Saved_models/tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)  
            def encode_text(tokenizer, lines, length):
                encoded = tokenizer.texts_to_sequences(lines)
                padded = pad_sequences(encoded, maxlen=length, padding='post')
                return padded
            
            estimate=model1.predict(encode_text(tokenizer,[review],length))[0][0]*5
            rating= round(estimate,1)
            
            if op!="real":
                messagebox.showinfo("Review Analysis","Classification: Not Suspicious\nSummary: "+summary+"\nRating: "+str(rating)+"/5.0")
                
            else:
                messagebox.showinfo("Review Analysis","Classification: Suspicious\nSummary: "+summary+"\nRating: "+str(rating)+"/5.0")
        
            
            
            
        self.Button4 = tk.Button(top)
        self.Button4.place(relx=0.417, rely=0.733, height=53, width=106)
        self.Button4.configure(activebackground="#ececec")
        self.Button4.configure(activeforeground="#000000")
        self.Button4.configure(background="#d9d9d9")
        self.Button4.configure(cursor="fleur")
        self.Button4.configure(disabledforeground="#a3a3a3")
        self.Button4.configure(font="-family {Segoe UI} -size 16 -weight bold")
        self.Button4.configure(foreground="#000000")
        self.Button4.configure(highlightbackground="#d9d9d9")
        self.Button4.configure(highlightcolor="black")
        self.Button4.configure(pady="0")
        self.Button4.configure(text='''GO''')
        self.Button4.configure(command=g4)


        self.Label2 = tk.Label(top)
        self.Label2.place(relx=0.683, rely=0.911, height=26, width=182)
        self.Label2.configure(background="#d9d9d9")
        self.Label2.configure(disabledforeground="#a3a3a3")
        self.Label2.configure(foreground="#000000")
        self.Label2.configure(text='''~Rohit Subramanian''')

        self.Label3 = tk.Label(top)
        self.Label3.place(relx=0.2, rely=0.089, height=56, width=352)
        self.Label3.configure(background="#d9d9d9")
        self.Label3.configure(disabledforeground="#a3a3a3")
        self.Label3.configure(font="-family {Georgia} -size 18 -weight bold")
        self.Label3.configure(foreground="#000000")
        self.Label3.configure(text='''Amazon Review Analysis''')

if __name__ == '__main__':
    vp_start_gui()





