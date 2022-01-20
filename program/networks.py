import sys
sys.path.append('..')
import os
import numpy as np
from tqdm.notebook import tqdm
import PIL
from PIL import Image
import pandas as pd
import pickle
import nltk
import torch
from torchvision.transforms.functional import to_pil_image
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EncoderCNN(nn.Module):
    """
    Encoder CNN Network for the encoding of images and generation of hidden state inputs for 
    RNN Decoder Model
    """
    def __init__(self, embed_size, pretrained=True):
        """
        
        arguments:
        --------------
        embed_size: int
            Size of the Word Embedding. (Vector Dimensionality of a Single Word)
            
        pretrained: bool
            If true uses pretrained CNN Model. Freezes all Convolutional and Pooling Layers of
            CNN Network.
        
        """
        super(EncoderCNN, self).__init__()
        self.pretrained = pretrained
        self.embed_size = embed_size
        resnet = models.resnet50(pretrained=pretrained)
        modules = list(resnet.children())[:-1]      
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        
    def forward(self, images):
        if self.pretrained:
            # Freezing of the Gradients does not create Graph Structure for autograd
            with torch.no_grad():
                features = self.resnet(images)
        else:
            # Creates Graph Structure
            features = self.resnet(images)
        # Reshape feature maps as a single feature vector
        features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        features = self.bn(features)
        
        return features
    

class DecoderRNN(nn.Module):
    """
    Decoder RNN Network for the encoding of image features and generation of Image Captioning    
    """
    def __init__(self, embed_size, hidden_size, vocab_size, 
                 verbose=False, **kwargs):
        """
        
        arguments:
        ---------------
        embed_size: int
            Dimensionality of the Word Embedding. This is used to init the Weigh Matrix for the inputs of the RNN
            
        hidden_size: int
            Dimensionality of the hidden Size. Dimensionality of the hidden states appended behind each other (unrolled)
            
        vocab_size: int
            Size of the vocabulary = Amount of Words given for training.
            
        num_layers; int
             Number of layers of the LSTM to stack on top of each other taking the hidden states as input from the previous layer.       
        """
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size 
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.verbose = verbose
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTMCell(embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dev = kwargs.get('device') if  kwargs.get('device') else self.__create_device()
        
    
    def __create_device(self):
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        
    def forward(self, features, captions, lengths):
        """
        Forward Prop with Teacher-Forcer Method. 
        Forces the Target to be the next input and not predicted label of previous.
        """
        hidden = None
        max_sequence_length = max(lengths)
        outputs = torch.empty((features.shape[0], captions.shape[1], self.vocab_size)).to(self.dev)
     
        # iteration N-1
        hidden, cell = self.lstm(features, hidden)
        feature_out = self.linear(hidden)
        outputs[:,0,:] = feature_out        
        
        captions = self.embed(captions)
        
        for t in range(max_sequence_length-1): # Not including last word as it need to be predicted by the previous
            # Step t used to predict step t+1
            inputs = captions[:, t, :]
            
            hidden, cell = self.lstm(inputs, (hidden, cell))
            out = self.linear(hidden)
            
            # Append each prediction to step t+1
            outputs[:, t+1, :] = out
                        
        return outputs.reshape(-1, self.vocab_size)
    
    
    def predict(self, features, max_sentence_length, end_token_id=1):
        predictions = []
        hidden = None
        
        # Iter N-1
        hidden, cell = self.lstm(features.unsqueeze(0), hidden)
        
        # Create Start Token as Tensor for first input
        inputs = torch.LongTensor([0])

        for t in range(max_sentence_length):
            inputs = self.embed(inputs)
            hidden, cell = self.lstm(inputs, (hidden, cell))
            out = self.linear(hidden)
            out = out.argmax(dim=1)
            if out == end_token_id:
                break
            predictions.append(out.item())
            inputs = out

        return predictions