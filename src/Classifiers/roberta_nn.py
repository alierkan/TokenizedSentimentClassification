# Importing the libraries needed
import random
from src import constants as con
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import seaborn as sns
import transformers
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
import logging
logging.basicConfig(level=logging.ERROR)

# Defining some key variables that will be used later on in the training
MAX_LEN = 256
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
LEARNING_RATE = 1e-05
path_dic = con.get_paths(con.SEMEVAL)
PATH = path_dic.get(con.LEMMA)
NUM_CLASSES = 3  # 3 for Positive, Negative, Neutral and 2 for Positive, Negative.
NUM_DIM = 768  # For Roberta-Base
EPOCHS = 1
PRETRAINED_ROBERTA = 'roberta-base'

tokenizer = RobertaTokenizer.from_pretrained(PRETRAINED_ROBERTA, truncation=True, do_lower_case=True)


class SentimentData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.Phrase
        self.targets = self.data.Sentiment
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


class MyReviews(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        print(self.filename)
        for line in open(self.filename, encoding="utf8"):
            yield line


def get_shuffle_list(file_pos, file_neg, file_neu, shuffle):
    xylist = []
    list_pos = MyReviews(file_pos)
    list_neg = MyReviews(file_neg)
    for t in list_pos:
        xylist.append((t, 0))
    for t in list_neg:
        xylist.append((t, 1))
    if file_neu:
        list_neu = MyReviews(file_neu)
        for t in list_neu:
            xylist.append((t, 2))
    if shuffle:
        random.shuffle(xylist)
    return xylist


class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained(PRETRAINED_ROBERTA)
        self.pre_classifier = torch.nn.Linear(NUM_DIM, NUM_DIM)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(NUM_DIM, NUM_CLASSES)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


def calcuate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct


def train(epoch, model, training_loader, loss_function, optimizer):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _, data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        outputs = model(ids, mask, token_type_ids)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accuracy(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if _ % 5000 == 0:
            loss_step = tr_loss / nb_tr_steps
            accu_step = (n_correct * 100) / nb_tr_examples
            print(f"Training Loss per 5000 steps: {loss_step}")
            print(f"Training Accuracy per 5000 steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct * 100) / nb_tr_examples}')
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")


def valid(model, testing_loader, loss_function):
    y_true, y_pred = [], []
    model.eval()
    n_correct = 0
    n_wrong = 0
    total = 0
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            outputs = model(ids, mask, token_type_ids).squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accuracy(big_idx, targets)

            y_true.extend(targets.tolist())
            y_pred.extend(big_idx.tolist())

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if _ % 5000 == 0:
                loss_step = tr_loss / nb_tr_steps
                accu_step = (n_correct * 100) / nb_tr_examples
                print(f"Validation Loss per 100 steps: {loss_step}")
                print(f"Validation Accuracy per 100 steps: {accu_step}")
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = n_correct / nb_tr_examples
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")

    with open(PATH + "RoBERTa_NN_scores_" + str(1) + ".txt", 'w') as outfile:
        for t, p in zip(y_true, y_pred):
            outfile.write(str(int(t)) + ";" + str(int(p)) + "\n")
    return epoch_accu


def main():
    if NUM_CLASSES == 3:
        train_list = get_shuffle_list(PATH + 'train-pos.txt',
                                      PATH + 'train-neg.txt',
                                      PATH + 'train-neu.txt', True)
        test_list = get_shuffle_list(PATH + 'test-pos.txt',
                                     PATH + 'test-neg.txt',
                                     PATH + 'test-neu.txt', False)
    elif NUM_CLASSES == 2:
        train_list = get_shuffle_list(PATH + 'train-pos.txt',
                                      PATH + 'train-neg.txt',
                                      None, True)
        test_list = get_shuffle_list(PATH + 'test-pos.txt',
                                     PATH + 'test-neg.txt',
                                     None, False)

    ###DATA
    train_data = pd.DataFrame(train_list)
    train_data.columns = ["Phrase", "Sentiment"]
    test_data = pd.DataFrame(test_list)
    test_data.columns = ["Phrase", "Sentiment"]

    print("TRAIN Dataset: {}".format(train_data.shape))
    print("TEST Dataset: {}".format(test_data.shape))

    training_set = SentimentData(train_data, tokenizer, MAX_LEN)
    testing_set = SentimentData(test_data, tokenizer, MAX_LEN)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                   'shuffle': False,
                   'num_workers': 0
                   }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    model = RobertaClass()
    model.to(device)

    # Creating the loss function and optimizer
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        train(epoch, model, training_loader, loss_function, optimizer)

    acc = valid(model, testing_loader, loss_function)
    print("Accuracy on test data = %0.2f%%" % acc)

    output_model_file = 'pytorch_roberta_sentiment.bin'
    output_vocab_file = './'

    model_to_save = model
    torch.save(model_to_save, output_model_file)
    tokenizer.save_vocabulary(output_vocab_file)

    print('All files saved')
    print('This tutorial is completed')


if __name__ == "__main__":
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    main()
