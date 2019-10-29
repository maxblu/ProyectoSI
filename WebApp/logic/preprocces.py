import spacy
from spacy.lang.es import Spanish , LOOKUP
from spacy.tokenizer import  Tokenizer
import os


class Lematize(object):

    def __init__ (self):
        nlp = Spanish()
        self.tokenizer = Tokenizer(nlp.vocab)
        self.stopwordsfolder ='data/stopwords/'

    def lematize(self,text):
        stopwords = []
        for file in os.listdir(self.stopwordsfolder):
            with open(self.stopwordsfolder + file, encoding='utf-8') as fd:
                stopwords += fd.read().split()
            
        doc = self.tokenizer(text)
        result = " "
        for word in doc:
            if word.text in stopwords:
                continue
            value =  LOOKUP.get(word.text)
            if value == None:
                result+= " " + word.text
                continue
            result += " " + value
        
        return  result