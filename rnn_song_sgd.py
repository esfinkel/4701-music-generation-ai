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

import ngram
from octave_vecs import vec_list_for_song as vectorizer

def _no_grad_normal_(tensor, mean, std):
    with torch.no_grad():
        return tensor.normal_(mean, std)

class RNN_No_FFNN(nn.Module):
    def __init__(self, hd_rnn, input_dim): # Add relevant parameters
        super(RNN_No_FFNN, self).__init__()
        self.h = hd_rnn 
        self.U = nn.Linear(hd_rnn, hd_rnn)    ## hidden input -> hidden layer
        self.V = nn.Linear(hd_rnn, input_dim) ## hidden layer -> out vector
        # self.V.weight = torch.nn.Parameter(self.custom_weights(5,2,hd_rnn,input_dim))
        self.W = nn.Linear(input_dim, hd_rnn) ## in vector -> hidden layer
        self.activation = nn.ReLU() 
        self.loss = nn.L1Loss() 

    # def custom_weights(self, mean, std, m, n):
    #     return torch.from_numpy(np.random.normal(mean, std, (m,n))).type(torch.FloatTensor)

    def init_hidden(self):
        return torch.zeros(self.h)

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)    

    def forward(self, line, hidden): 
        tensor = torch.from_numpy(line).float() 
        h_t = self.activation(self.U(hidden) + self.W(tensor))
        return self.V(h_t), h_t

def load_and_vectorize_data(directory):
    """Load all of the kern files from the directory; vectorize all data;
    return vectors with gold labels.
    In particular, this will return a list of elements (note_a, note_b). 
    note_a and note_b are representations of sequential notes of a song
    (the songs are all concatted together, with two zero vectors between
    each two adjacent songs).
    Each time slice is represented as a vector of length 600; each 100 indices 
    represents the pitch/duration of one note. If a time slice has >6 notes,
    some are ignored.
    At each step, note_a will be the context, and note_b will be the 
    gold label; the RNN will try to predict note_b from note_a.
    """
    # directory should be "music_in_C_training"
    veclist = []
    for filename in os.listdir(f"./{directory}"):
        if ".DS_Store" in filename:
            continue
        with open(f"./{directory}/{filename}", "r") as f:
            filetext = f.readlines()
        file_vec = vectorizer(filetext)
        # veclist.append(vectorizer(filetext))
        filevec_with_labels = []
        for i in range(len(file_vec)-1): # for each time slice
            if isinstance(file_vec[i], np.ndarray):
                filevec_with_labels.append((file_vec[i], file_vec[i+1]))
        veclist.append(filevec_with_labels)
    return veclist


def main(hidden_dim, num_epochs, learning_rate, existing_model=None): 
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
        'Song_GD',
        'octave_vecs',
        'hidden_dim='+str(hidden_dim),
        'learning_rate='+str(learning_rate)
    ]
    partial_file_name= 'rnn_models/'+'&'.join(file_partial_name)
    min_valid_dist = None

    prev_val_dist = 0 
    has_gone_down = False 
    for epoch in range(num_epochs):
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

                ## TODO: try only computing loss every few notes? 
                total += 1 
                song_length += 1
                if hidden is None:
                    hidden = model.init_hidden()
                predicted_next_line, hidden = model(line, hidden) 
                tot_distance += torch.linalg.norm(predicted_next_line - torch.from_numpy(gold_next_line))
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
                tot_distance += torch.linalg.norm(pred_next_line - torch.from_numpy(gold_next_line))

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
        # if (val_acc - prev_val_accuracy < 0 and has_gone_down):
        if tot_distance / total  > prev_val_dist and has_gone_down:
            print("***WARNING*** OVERFITTING")
            # break 
        elif tot_distance / total > prev_val_dist:
            has_gone_down = True
        else:
            has_gone_down = False 
        prev_val_dist = tot_distance / total 


if __name__ == "__main__":
    hidden_dim_rnn = 30 ##TODO: try other values
    number_of_epochs = 20
    lr = 0.25 ##TODO: try other values 
    # with open('./rnn_models/Song_GD&num_note_vecs&hidden_dim=200&learning_rate=0.4&epoch=1&dist=17.862510681152344', 'rb') as f:
    #     model = torch.load(f)
    main(hidden_dim=hidden_dim_rnn, num_epochs=number_of_epochs, learning_rate=lr, existing_model=None)

