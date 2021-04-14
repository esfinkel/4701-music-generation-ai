"""
Like rnn_song_sgd, but more classification-oriented. Number of notes is
one-hot instead of integer; trying classification-oriented setup, including
loss function `torch.nn.KLDivLoss`.
"""

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
import math
from tqdm import tqdm

import ngram
from probability_vectors import vec_list_for_song as vectorizer

def _no_grad_normal_(tensor, mean, std):
    with torch.no_grad():
        return tensor.normal_(mean, std)

class RNN_No_FFNN(nn.Module):
    def __init__(self, hd_rnn, input_dim): # Add relevant parameters
        super(RNN_No_FFNN, self).__init__()
        self.h = hd_rnn 
        self.U = nn.Linear(hd_rnn, hd_rnn)    ## hidden input -> hidden layer
        self.V = nn.Linear(hd_rnn, input_dim) ## hidden layer -> out vector
        self.W = nn.Linear(input_dim, hd_rnn) ## in vector -> hidden layer
        self.activation = nn.Sigmoid()
        self.softmax = nn.LogSoftmax()
        self.loss = torch.nn.KLDivLoss() # also tried nn.BCELoss(), nn.MSELoss()


    def init_hidden(self):
        return torch.zeros(self.h)

    def compute_Loss(self, predicted_vector, gold_label):
        softmax = nn.LogSoftmax()
        gold_label = gold_label.float()
        ## DON'T TAKE LOG
        soft_gold = torch.cat((
            torch.log_softmax(gold_label[:13], 0), #pitch 1 L
            torch.log_softmax(gold_label[13:26], 0), #pitch 2 L
            torch.log_softmax(gold_label[26:39], 0), #pitch 3 L
            torch.log_softmax(gold_label[39:48], 0), #duration L
            torch.log_softmax(gold_label[48:52], 0), #num notes L 
            torch.log_softmax(gold_label[52:13+52], 0), #pitch 1 R
            torch.log_softmax(gold_label[13+52:26+52], 0), #pitch 2 R
            torch.log_softmax(gold_label[26+52:39+52], 0), #pitch 3 R
            torch.log_softmax(gold_label[39+52:48+52], 0), #duration R
            torch.log_softmax(gold_label[48+52:52+52], 0) #num notes R
        ))
        return self.loss(predicted_vector, gold_label.float())   

    def forward(self, line, hidden): 
        tensor = torch.from_numpy(line).float() 
        h_t = self.activation(self.U(hidden) + self.W(tensor))
        pred_vector = self.V(h_t)
        pred_logsoft = torch.cat((
            torch.log_softmax(pred_vector[:13], 0), #pitch 1 L
            torch.log_softmax(pred_vector[13:26], 0), #pitch 2 L
            torch.log_softmax(pred_vector[26:39], 0), #pitch 3 L
            torch.log_softmax(pred_vector[39:48], 0), #duration L
            torch.log_softmax(pred_vector[48:52], 0), #num notes L 
            torch.log_softmax(pred_vector[52:13+52], 0), #pitch 1 R
            torch.log_softmax(pred_vector[13+52:26+52], 0), #pitch 2 R
            torch.log_softmax(pred_vector[26+52:39+52], 0), #pitch 3 R
            torch.log_softmax(pred_vector[39+52:48+52], 0), #duration R
            torch.log_softmax(pred_vector[48+52:52+52], 0) #num notes R
        ))
        return pred_logsoft.float(), h_t

def load_and_vectorize_data(directory):
    veclist = []
    for filename in os.listdir(f"./{directory}"):
        if ".DS_Store" in filename:
            continue
        with open(f"./{directory}/{filename}", "r") as f:
            filetext = f.readlines()
        file_vec = vectorizer(filetext)
        filevec_with_labels = []
        for i in range(len(file_vec)-1): 
            if isinstance(file_vec[i], np.ndarray):
                filevec_with_labels.append((file_vec[i], file_vec[i+1]))
        veclist.append(filevec_with_labels)
    return veclist

def main(hidden_dim, num_epochs, learning_rate, existing_model=None, epoch_start=0): 
    print("Fetching and vectorizing data")
    train_data = load_and_vectorize_data("music_in_C_training") 
    valid_data = load_and_vectorize_data("music_in_C_test")
    print("Fetched and vectorized data")

    if existing_model is None:
        model = RNN_No_FFNN(hidden_dim, len(train_data[0][0][0]))
    else:
        model = existing_model
    optimizer = optim.SGD(model.parameters(),lr=learning_rate, momentum=0.9) 

    file_partial_name = [
        'log_prob_vecs',
        'hidden_dim='+str(hidden_dim),
        'learning_rate='+str(learning_rate)
    ]
    partial_file_name= 'rnn_models/'+'&'.join(file_partial_name)

    min_valid_dist = None
    prev_val_dist = 0 
    dist_has_increased = False 
    for epoch in range(epoch_start, epoch_start+num_epochs):
        model.train()
        optimizer.zero_grad()

        loss = torch.Tensor([0])
        tot_loss = torch.Tensor([0])
        tot_distance = 0
        total = 0
        song_length = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        for song in tqdm(train_data): 
            hidden = None
            for line, gold_next_line in song:
                total += 1 
                song_length += 1
                if hidden is None:
                    hidden = model.init_hidden()
                predicted_next_line, hidden = model(line, hidden) 
                tot_distance += torch.linalg.norm(math.e**predicted_next_line - torch.from_numpy(gold_next_line))
                curr_loss = model.compute_Loss(predicted_next_line, torch.from_numpy(gold_next_line))
                loss += curr_loss 
                tot_loss += curr_loss
            optimizer.zero_grad()
            loss = loss / song_length 
            song_length = 0
            loss.backward()
            optimizer.step()
            loss = torch.Tensor([0])

        loss_avg = tot_loss.item() / total 
        print(f"Average Loss: {loss_avg}")
 
        ### validation 
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training avg distance for epoch {}: {}".format(epoch + 1, tot_distance / total))
        print("Training time for this epoch: {}".format(time.time() - start_time))

        tot_distance = 0
        total = 0
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        random.shuffle(valid_data)

        for song in valid_data:
            hidden = None
            for line, gold_next_line in song:
                if hidden is None:
                    hidden = model.init_hidden()
                pred_next_line, hidden = model(line, hidden)
                total += 1
                tot_distance += torch.linalg.norm(math.e**pred_next_line - torch.from_numpy(gold_next_line))

        if min_valid_dist is None or tot_distance / total < min_valid_dist:
            min_valid_dist = tot_distance / total 
            ## save best model
            with open(f"{partial_file_name}&epoch={epoch+1}&dist={min_valid_dist}", "wb") as f:
                torch.save(model, f)

        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation distance for epoch {}: {}".format(epoch + 1, tot_distance / total))
        print("Validation time for this epoch: {}".format(time.time() - start_time))

        ##STOPPING CONDITION 
        # val_acc = correct / total 
        if tot_distance / total  > prev_val_dist and dist_has_increased:
            print("***WARNING*** OVERFITTING")
            # break 
        elif tot_distance / total > prev_val_dist:
            dist_has_increased = True
        else:
            dist_has_increased = False 
        prev_val_dist = tot_distance / total 


if __name__ == "__main__":
    hidden_dim_rnn = 50
    number_of_epochs = 20
    lr = 0.25
    # with open('rnn_models/Song_GD&octave_vecs&hidden_dim=30&learning_rate=0.25&1618103280&epoch=20&dist=12.615211486816406', 'rb') as f:
    #     model = torch.load(f)
    main(hidden_dim=hidden_dim_rnn, num_epochs=number_of_epochs, learning_rate=lr, existing_model=None, epoch_start=1)

