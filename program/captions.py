import pandas as pd
from multiprocess import Pool
import re
from collections import Counter
from tqdm.notebook import tqdm
import numpy as np
import torch


class TextPreprocessor(object):
    """Class handles text data."""
    
    def __init__(self, n_processes, max_sequence_length=None,
                 start_token='<START>', end_token='<END>', fill_token='<PAD>', unknown_token='<UNK>', 
                 pad_fill=True, disable_pbar=True):
        """
        Handles text from csvs, extract tokens with padding or without. Can create a dictionary with words
        to idx.        
        """
        self.n_processes = n_processes
        self.max_sequence_length = max_sequence_length
        self.start_token = start_token
        self.end_token = end_token
        self.fill_token = fill_token
        self.pad_fill = pad_fill
        self.unknown_token = unknown_token
        self.disable_pbar = disable_pbar
        self.longest_word = None
        self.tokenized_text = None
        self.text_dict = None
        self.text_dict_r = None
    
    @staticmethod
    def _longest_word(X):
        """Extracts longest word from all given sentences"""
        X_ = pd.Series(X)
        sentence_lengths = X_.map(lambda x: len(x.split(' ')))
        longest_word = max(sentence_lengths)
        
        return (sentence_lengths, longest_word)

    def tokenize(self, X, return_tokenized=False):
        """Tokenizes Sentence by the help of multiprocessing"""
        self.sentence_lengths, self.longest_word = self._longest_word(X=X)
        if self.max_sequence_length:
            # Overwrite props
            self.longest_word = self.max_sequence_length
        
        def __tokenize_words(s):
            s = re.sub(pattern=r'[^\w\s]', repl='', string=s)
            s = s.lower()
            #s = s.strip()
            w = s.split(' ')  
            if len(w) > self.longest_word:
                w = w[:self.longest_word]
            w.insert(0, self.start_token)
            w.append(self.end_token)
            if self.pad_fill:
                while len(w)-2 < self.longest_word:
                    w.append('<PAD>')
            
            return w
        
        with Pool(processes=self.n_processes) as pool:
            tokenized_text = pool.map(__tokenize_words, X)
        
        self.tokenized_text = tokenized_text
        
        if return_tokenized:
            return tokenized_text
    
    def create_txt_dict(self):
        """Creates a dictionary object which can be used to translate from word to idx"""
        self.text_dict = {self.start_token:1, self.end_token:2, self.unknown_token:3, self.fill_token:4}
        idx = 5

        for text in self.tokenized_text:
            for word in text:
                if not self.text_dict.get(word):
                    self.text_dict[word] = idx
                    idx += 1
        
        self.zero_index()
            
        return self
    
    def zero_index(self):
        val_range = np.arange(len(list(self.text_dict.values())))
        dict_keys = list(self.text_dict.keys())
        self.text_dict = dict(zip(dict_keys, val_range))
    
    def idx_to_word(self, idx: list):
        """Creates a reverse dictionary which can be used to translate from idx back to word"""
        if not self.text_dict:
            self.create_txt_dict()
        
        # Create dict to transform words back
        if not self.text_dict_r:
            self.text_dict_r = dict(zip(list(self.text_dict.values()), list(self.text_dict.keys())))
        
        return ' '.join([self.text_dict_r[id_] for id_ in idx.tolist()])
    
    def word_to_idx(self, word):
        """Takes a word and returns its index"""
        if not self.text_dict:
            self.create_txt_dict()
        
        return self.text_dict[word]
    
    def count_words(self):
        """Counts length of tokenized sentences."""
        concat_lists = []

        for L in tqdm(self.tokenized_text, disable=self.disable_pbar):
            for element in L:
                concat_lists.append(element)
                
        return Counter(concat_lists)
    
class TextHandler(TextPreprocessor):
    
    def __init__(self, n_processes: int, max_sequence_length: int =None, unknown_threshold: float = None, start_token: str ='<START>', 
                 end_token: str ='<END>', fill_token: str='<PAD>', unknown_token: str ='<UNK>', pad_fill: bool =True, disable_pbar=True):
        """
        
        arguments:
        ------------------
        n_processes: int
            Amount processed to use for the tokenization
            
        unknown_threshold: int or float
            Less frequent words to remove from data
            
        start_token: str
            default: '<START>' Start token
        
        end_token: str
            default: '<END>', End token
        
        fill_token: str 
            default:'<PAD>', Pad token
        
        unknown_token: str
            default:'<UNK>', unknown token
        
        pad_fill: bool
            default=True, add_idx_for_new_word=False
        """
        super().__init__(n_processes=n_processes, max_sequence_length=max_sequence_length,
                         start_token=start_token, end_token=end_token, 
                         fill_token=fill_token, unknown_token=unknown_token, pad_fill=pad_fill, 
                         disable_pbar=disable_pbar)
        self.unknown_threshold = unknown_threshold
        self.disable_pbar = disable_pbar
        
        
    def fit(self, X: list):
        """"""
        self.tokenize(X, return_tokenized=False)
        self.create_txt_dict()
        if self.unknown_threshold:
            self._tokenize_unknown()
        
        return self
        
    def fit_transform(self, X: list):
        """"""
        self.fit(X)
        X = self.transform(X)
        
        return X
    
    def transform(self, X: list):
        """"""
        converted_tensors = []
        for s in tqdm(X, desc='Converting Sentences to Tensors', disable=self.disable_pbar):
            converted_sentence = self.convert_words_to_tensor(sentence=s)
            converted_tensors.append(converted_sentence)
            
        if len(converted_tensors)==1:
            converted_tensors = converted_tensors[0]
        
        return converted_tensors
    
    def inverse_transform(self, X: list):
        converted_sentences = []
        for t in tqdm(X, desc='Converting Tensors to Sentences', disable=self.disable_pbar):
            sentence = self.idx_to_word(t)
            converted_sentences.append(sentence)
        
        return converted_sentences
        
    def _tokenize_unknown(self):
        """Replaces less appearing words with unknown token"""
        word_counts = pd.Series(self.count_words())
        words = word_counts[word_counts > self.unknown_threshold].index.to_list()
        
        self.text_dict = {self.start_token:1, self.end_token:2, self.unknown_token:3, self.fill_token:4}
        token = 5
        for w in words:
            if w not in list(self.text_dict.keys()):
                self.text_dict[w] =  token
                token += 1
        self.text_dict_r = None
        
        self.zero_index()

        return self
        
    def convert_words_to_tensor(self, sentence: str):
        """Converts given sentence into a tensor."""
        if not self.text_dict:
            self.create_txt_dict()
            
        # Inner func for tokenization
        def __tokenize_words(s):
            s = re.sub(pattern=r'[^\w\s]', repl='', string=s)
            s = s.lower()
            #s = s.strip()
            w = s.split(' ')  
            if len(w) > self.longest_word:
                w = w[:self.longest_word]
            w.insert(0, self.start_token)
            w.append(self.end_token)
            if self.pad_fill:
                while len(w)-2 < self.longest_word:
                    w.append('<PAD>')
            
            return w
        
        tensor_convert = []
        for w in __tokenize_words(sentence):
            try:
                tensor_convert.append(self.text_dict[w])
            except:
                # if word is not in dict
                tensor_convert.append(self.text_dict[self.unknown_token])
                
        return torch.LongTensor(tensor_convert)