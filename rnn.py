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
from kern_to_vec2 import vec_list_for_song as vectorizer

class RNN_No_FFNN(nn.Module):
    def __init__(self, hd_rnn, input_dim): # Add relevant parameters
        super(RNN_No_FFNN, self).__init__()
        self.h = hd_rnn 
        self.U = nn.Linear(hd_rnn, hd_rnn) 
        self.V = nn.Linear(hd_rnn, input_dim) 
        self.W = nn.Linear(input_dim, hd_rnn) 
        self.activation = nn.ReLU() 

        self.softmax = nn.LogSoftmax()
        # self.loss = nn.NLLLoss()
        self.loss = nn.L1Loss() 

        self.prev_vec = torch.zeros(self.h)

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)    

    def forward(self, inputs): 
        '''
        takes in list of word embedding vectors, returns predicted 
        label probability distribution as single vector 
        '''
        tensor = torch.from_numpy(inputs).float() 
        h_prev = self.prev_vec #torch.zeros(self.h)
        # h_t = 0 
        # for word in inputs:
        h_t = self.activation(self.U(h_prev) + self.W(tensor))
        h_prev = h_t 
        self.prev_vec = h_prev
        # return self.softmax(self.V(h_t))
        return self.V(h_t)


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
            if isinstance(file_vec[i], np.ndarray):# and sum(veclist[i])!=0:
                filevec_with_labels.append((file_vec[i], file_vec[i+1]))
        veclist.append(filevec_with_labels)
    return veclist
    # possibly store vectors to disk if it takes longer to make the
    # vectors than to retrieve them, but idk if it will

def load_and_vectorize_data_indexed(directory):
    """Same as load_and_vectorize_data, but returns a list of elements
    (i, note_a, note_b), where i is an index into the data. """
    veclist = load_and_vectorize_data(directory)
    veclist_indices = []
    for i in range(len(veclist)):
        song = veclist[i]
        veclist_indices.append((i, song))
    return veclist_indices


def main(hidden_dim, num_epochs, learning_rate): # Add relevant parameters
    print("Fetching and vectorizing data")
    train_data = load_and_vectorize_data("music_in_C_training") # would incorporate k to get subset
    # valid_data = load_and_vectorize_data_indexed("music_in_C_test")
    valid_data = load_and_vectorize_data("music_in_C_test")
    print("Fetched and vectorized data")

    # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
    # Further, think about where the vectors will come from. There are 3 reasonable choices:
    # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
    # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
    # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further. 
    # Option 3 will be the most time consuming, so we do not recommend starting with this


    model = RNN_No_FFNN(hidden_dim, len(train_data[0][0][0]))
    optimizer = optim.SGD(model.parameters(),lr=learning_rate, momentum=0.9) # This network is trained by traditional (batch) gradient descent; ignore that this says 'SGD'

    file_partial_name = [
        str(int(time.time())),
        'hidden_dim='+str(hidden_dim),
        'num_epochs='+str(num_epochs),
        'learning_rate='+str(learning_rate)
    ]
    partial_file_name= 'rnn_models/'+'&'.join(file_partial_name)
    # max_valid_acc = 0
    min_valid_dist = None


    # while not stopping_condition(x): # How will you decide to stop training and why
    ##STOPPING CONDITION: check if validation accuracy goes down two times in a row 
    ## OR we've reached the number of epochs 
    prev_val_dist = 0 
    has_gone_down = False 
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        loss = torch.Tensor([0])
        # correct = 0
        tot_distance = 0
        total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        for song in tqdm(train_data): ## list of word embed vecs, labels 
            for line, gold_next_line in song:
                predicted_next_line = model(line) 
                # predicted_label = torch.argmax(predicted_vector)
                # correct += int(predicted_label == gold_label)
                total += 1 
                tot_distance += torch.linalg.norm(predicted_next_line - torch.from_numpy(gold_next_line))
                # loss += model.compute_Loss(predicted_next_line.view(1,-1), torch.tensor([gold_next_line])) 
                loss += model.compute_Loss(predicted_next_line, torch.from_numpy(gold_next_line))
            model.prev_vec = torch.zeros(model.h)
        # loss = loss / len(train_data) 
        loss = loss / total
        loss_float = loss.item()
        print(loss_float)
        print("Backpropagation...")
        loss.backward() 
        print("Finished backpropagation, beginning optimization step")
        optimizer.step()
 

        ### validation 
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training avg distance for epoch {}: {}".format(epoch + 1, tot_distance / total))
        print("Training time for this epoch: {}".format(time.time() - start_time))

        # correct = 0
        tot_distance = 0
        total = 0
        valid_file_string = 'idx,prediction,gold_label\n'
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        random.shuffle(valid_data)

        # for idx, input_stream, gold_label in valid_data:
        #     predicted_vector = model(input_stream)
        #     predicted_label = torch.argmax(predicted_vector)
        #     correct += int(predicted_label == gold_label)
        #     total += 1
        #     valid_file_string += (
        #         str(idx)+','+str(int(predicted_label.item()))+','+str(gold_label)+'\n'
        #     )
        for song in valid_data:
            for line, gold_next_line in song:
                pred_next_line = model(line)
                # predicted_label = torch.argmax(predicted_vector)
                # correct += int(predicted_label == gold_label)
                total += 1
                tot_distance += torch.linalg.norm(pred_next_line - torch.from_numpy(gold_next_line))
                # valid_file_string += (
                #     str(idx)+','+str(int(predicted_label.item()))+','+str(gold_label)+'\n'
                # )
            model.prev_vec = torch.zeros(model.h)

        # if (correct/total) > max_valid_acc:
        if min_valid_dist is None or tot_distance / total < min_valid_dist:
            # max_valid_acc = correct / total
            min_valid_dist = tot_distance / total 
            with open(f"{partial_file_name}&epoch={epoch+1}&dist={min_valid_dist}", "wb") as f:
                torch.save(model, f)

            # with open(
            #     partial_file_name+'&epoch={:d}&loss={:.3f}.csv'.format(epoch+1, loss_float),
            #     'w') as f:
            #     f.write(valid_file_string)


        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation distance for epoch {}: {}".format(epoch + 1, tot_distance / total))
        print("Validation time for this epoch: {}".format(time.time() - start_time))

        ##STOPPING CONDITION 
        # val_acc = correct / total 
        # if (val_acc - prev_val_accuracy < 0 and has_gone_down):
        if tot_distance / total  > prev_val_dist and has_gone_down:
            print("Overfitting. Stopping.")
            # break 
        elif tot_distance / total > prev_val_dist:
            has_gone_down = True
        else:
            has_gone_down = False 
        prev_val_dist = tot_distance / total 

        # END ADDED CODE 
        # You may find it beneficial to keep track of training accuracy or training loss; 

        # Think about how to update the model and what this entails. Consider ffnn.py and the PyTorch documentation for guidance

        # You will need to validate your model. All results for Part 3 should be reported on the validation set. 
        # Consider ffnn.py; making changes to validation if you find them necessary

if __name__ == "__main__":
    hidden_dim_rnn = 200
    number_of_epochs = 20
    lr = 0.25
    main(hidden_dim=hidden_dim_rnn, num_epochs=number_of_epochs, learning_rate=lr)

