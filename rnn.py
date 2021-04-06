import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import io
import pickle

import time
from tqdm import tqdm
from data_loader import fetch_data

unk = '<UNK>'

# use the pytorch RNN layer

class RNN(nn.Module):
    def __init__(self, hd_rnn, input_dim, to_ffnn, hd_ffnn): # Add relevant parameters
        super(RNN, self).__init__()
        self.h = hd_rnn 
        self.U = nn.Linear(hd_rnn, hd_rnn) 
        self.V = nn.Linear(hd_rnn, to_ffnn) 
        self.W = nn.Linear(input_dim, hd_rnn) 
        self.activation = nn.ReLU() 
        # Fill in relevant parameters
        # Ensure parameters are initialized to small values, see PyTorch documentation for guidance
        self.softmax = nn.LogSoftmax()
        self.loss = nn.NLLLoss() 
        ##FFNN PIECE
        self.W1 = nn.Linear(to_ffnn, hd_ffnn) 
        self.W2 = nn.Linear(hd_ffnn, 5) 

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)    

    def forward(self, inputs): 
        '''
        takes in list of word embedding vectors, returns predicted 
        label probability distribution as single vector 
        '''
        h_prev = torch.zeros(self.h)
        h_t = 0 
        for word in inputs:
            h_t = self.activation(self.U(h_prev) + self.W(word))
            h_prev = h_t 
        y = self.activation(self.V(h_t))
        ##FINAL FFNN 
        z1 = self.activation(self.W1(y))
        z2 = self.W2(z1)
        predicted_vector = self.softmax(z2)
        return predicted_vector

class RNN_No_FFNN(nn.Module):
    def __init__(self, hd_rnn, input_dim): # Add relevant parameters
        super(RNN_No_FFNN, self).__init__()
        self.h = hd_rnn 
        self.U = nn.Linear(hd_rnn, hd_rnn) 
        self.V = nn.Linear(hd_rnn, 5) 
        self.W = nn.Linear(input_dim, hd_rnn) 
        self.activation = nn.ReLU() 
        # Fill in relevant parameters
        # Ensure parameters are initialized to small values, see PyTorch documentation for guidance
        self.softmax = nn.LogSoftmax()
        self.loss = nn.NLLLoss() 

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)    

    def forward(self, inputs): 
        '''
        takes in list of word embedding vectors, returns predicted 
        label probability distribution as single vector 
        '''
        h_prev = torch.zeros(self.h)
        h_t = 0 
        for word in inputs:
            h_t = self.activation(self.U(h_prev) + self.W(word))
            h_prev = h_t 
        return self.softmax(self.V(h_t))


## WORD EMBEDDINGS BEGIN 
def load_pretrained_embedding_vectors(fname):
	# source for the embeddings:
	# 	https://fasttext.cc/docs/en/english-vectors.html
	# vectors available thanks to
	# 	T. Mikolov, E. Grave, P. Bojanowski, C. Puhrsch, A. Joulin.
	# 	Advances in Pre-Training Distributed Word Representations
	# 	https://arxiv.org/abs/1712.09405
	data = {}
	with io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
		is_first = True
		for line in tqdm(fin):
			if is_first:
				is_first = False 
				continue
			tokens = line.rstrip().split(' ')
			data[tokens[0]] = list(map(float, tokens[1:]))
	with open('pretrained_vectors/wiki-news-300d-1M-pickle.p', 'wb') as f:
		pickle.dump(data, f)
	return data

def load_fastText_vectors():
	with open('pretrained_vectors/wiki-news-300d-1M-pickle.p', 'rb') as f:
		return pickle.load(f)

## WORD EMBEDDINGS END 


def convert_input_to_vectors(data, embed):
    ''' 
    takes in list of (doc, label), 
    returns list of (list of word embedding vectors, label)
    '''
    vectorized_data = [] 
    # tot_count = 0
    # unk_count = 0
    for document, y in data:
        vec_list = []
        for word in document:
            word = word.strip()
            # tot_count += 1
            # if word not in embed:
                # print(word)
                # unk_count += 1
            vec_list.append(torch.Tensor(embed.get(word, [0]*300)))
        vectorized_data.append((vec_list,y))
    # print('prop unknown: {:.2f}'.format(unk_count/tot_count))
    return vectorized_data

def convert_input_to_vectors_indexed(data, embed):
    ''' 
    takes in list of (idx, doc, label), 
    returns list of (list of word embedding vectors, label)
    '''
    vectorized_data = [] 

    for idx, document, y in data:
        vec_list = []
        for word in document:
            word = word.strip()
            vec_list.append(torch.Tensor(embed.get(word, [0]*300)))
        vectorized_data.append((idx, vec_list,y))
    return vectorized_data

def main(hidden_dim, to_ffnn, hd_ffnn, num_epochs, learning_rate, k, ffnn): # Add relevant parameters
    print("Fetching data")
    train_data, valid_data = fetch_data(k) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

    # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
    # Further, think about where the vectors will come from. There are 3 reasonable choices:
    # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
    # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
    # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further. 
    # Option 3 will be the most time consuming, so we do not recommend starting with this

    print("Fetched data")
    print("fetching embeddings")
    embed = load_fastText_vectors()
    # embed = {'the' : [0]*300}
    print("fetched embeddings, vectorizing data....")
    train_data = convert_input_to_vectors(train_data, embed)
    valid_data = convert_input_to_vectors_indexed(valid_data, embed)
    print("Vectorized data")
    if ffnn:
        model = RNN(hidden_dim, len(embed['the']), to_ffnn, hd_ffnn)
    else:
        model = RNN_No_FFNN(hidden_dim, len(embed['the']))
    optimizer = optim.SGD(model.parameters(),lr=learning_rate, momentum=0.9) # This network is trained by traditional (batch) gradient descent; ignore that this says 'SGD'
    # orig lr = 0.01, momentum = 0.9

    file_partial_name = [
        str(int(time.time())),
        'hidden_dim='+str(hidden_dim),
        'to_ffnn='+str(to_ffnn),
        'hd_ffnn='+str(hd_ffnn),
        'num_epochs='+str(num_epochs),
        'learning_rate='+str(learning_rate),
        'datasize='+str(k),
    ]
    partial_file_name= 'rnn_data/'+'&'.join(file_partial_name)
    max_valid_acc = 0


    # while not stopping_condition(x): # How will you decide to stop training and why
    ##STOPPING CONDITION: check if validation accuracy goes down two times in a row 
    ## OR we've reached the number of epochs 
    prev_val_accuracy = 0 
    has_gone_down = False 
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        loss = torch.Tensor([0])
        correct = 0
        total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        for input_stream, gold_label in tqdm(train_data): ## list of word embed vecs, labels 
            predicted_vector = model(input_stream)
            predicted_label = torch.argmax(predicted_vector)
            correct += int(predicted_label == gold_label)
            total += 1 
            loss += model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label])) 
        loss = loss / len(train_data) 
        loss_float = loss.item()
        print(loss_float)
        print("Backpropagation...")
        loss.backward() 
        print("Finished backpropagation, beginning optimization step")
        optimizer.step()
    # START ADDED CODE

        # 1. initialize
        # 2. call forward() - new nodes created for inputs
        # 3. in forward(), apply fxns - dynamically added to graphs
        #      construct + evaluate computation graph
        # 4. compute loss; entire graph build, we have done forward pass
        # 5. compute gradients for each weight in loss.backward() - backprop step
        #     variable.gradient() ?    cumulative?
        # 6. update weights using the gradients; optimizer.step()
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Training time for this epoch: {}".format(time.time() - start_time))

        correct = 0
        total = 0
        valid_file_string = 'idx,prediction,gold_label\n'
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        random.shuffle(valid_data)

        for idx, input_stream, gold_label in valid_data:
            predicted_vector = model(input_stream)
            predicted_label = torch.argmax(predicted_vector)
            correct += int(predicted_label == gold_label)
            total += 1
            valid_file_string += (
                str(idx)+','+str(int(predicted_label.item()))+','+str(gold_label)+'\n'
            )

        if (correct/total) > max_valid_acc:
            max_valid_acc = correct / total
            with open(
                partial_file_name+'&epoch={:d}&loss={:.3f}.csv'.format(epoch+1, loss_float),
                'w') as f:
                f.write(valid_file_string)


        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Validation time for this epoch: {}".format(time.time() - start_time))

        ##STOPPING CONDITION 
        val_acc = correct / total 
        if (val_acc - prev_val_accuracy < 0 and has_gone_down):
            print("Overfitting. Stopping.")
            break 
        elif val_acc - prev_val_accuracy < 0:
            has_gone_down = True
        else:
            has_gone_down = False 
        prev_val_accuracy = val_acc 

        # END ADDED CODE 
        # You may find it beneficial to keep track of training accuracy or training loss; 

        # Think about how to update the model and what this entails. Consider ffnn.py and the PyTorch documentation for guidance

        # You will need to validate your model. All results for Part 3 should be reported on the validation set. 
        # Consider ffnn.py; making changes to validation if you find them necessary

if __name__ == "__main__":
    # d = load_fastText_vectors()
    # print(d['the'])
    pass
