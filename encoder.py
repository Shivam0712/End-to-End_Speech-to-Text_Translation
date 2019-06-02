import numpy as np
import csv
import string
import re
import pandas as pd
import pickle

class encoder():
    def __init__(self, listOfSentences):
        
        ## Create Dictionary of all unique words in the listOfSentences
        self.dictionary, self.max_size = createDictionary(listOfSentences)
        
        ## Length of vocabulary
        self.vocab_size = len(list(self.dictionary.keys()))
        
    def encode(self, text):
        text = '<s> '+ text +' </s>'
        outs = []
        for word in text.split(' '):
            word = word.strip()
            if (word != "") & (word in list(self.dictionary.keys())):
                hot = np.zeros(self.vocab_size)
                hot[self.dictionary[word]] = 1
                outs.append(hot)
        if len(outs) < self.max_size + 2:
            for i in range(self.max_size + 2 - len(outs)):
                hot = np.zeros(self.vocab_size)
                hot[self.dictionary['_']] = 1
                outs.append(hot)
        if len(outs) > self.max_size + 2:
            outs = outs[0: self.max_size + 2]
        return np.vstack(outs)

def createDictionary(listOfSentences):
    wordsDB = {'<s>':0, '</s>':1, '_': 2}
    counter = 3
    max_len = 0
    for line in listOfSentences:
        ln_len = 0
        for word in line.split(' '):
            word = word.strip()
            if (word != ""):
                ln_len = ln_len + 1
                if word not in list(wordsDB.keys()): 
                    wordsDB[word] = counter
                    counter = counter + 1
        if ln_len > max_len:
            max_len = ln_len
    return wordsDB, max_len

