import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
import utils
import time

DATA_SIZE = 8192
BATCH_SIZE = 32
NCOLS = 300
NCOLS_MUL = 2
NWIN = 5
NUM_EPOCHS = 20
NUM_CLASSES = 1
NUM_FILTERS = 600
NUM_TRIALS = 5
KERNEL_SIZE = (4, 600)
PATH = "/../"


def get_review_windows(vocab_txt_to_id, reviews, max_rlen, index, index_limit):
    data_size = DATA_SIZE if index < index_limit else (len(reviews) - DATA_SIZE * index_limit)
    x = np.zeros(shape=(data_size, max_rlen))
    y = np.zeros(shape=(data_size, NUM_CLASSES))
    start_index = index * DATA_SIZE
    for i, review in enumerate(reviews[start_index: start_index + data_size]):
        for j, vocab in enumerate(review[0]):
            x[i][j] = vocab_txt_to_id.get(vocab)
        y[i] = review[1]
    x = x.astype('float32')
    return x, y


def get_pretrained(vocab_txt_to_id, model, vocabs, glove_dict):
    x = np.zeros(shape=(len(vocabs), NCOLS_MUL*NCOLS))
    for vocab in vocabs:
        try:
            wv = np.asarray(utils.get_wordvector(model, vocab))
            if NCOLS_MUL > 1:
                glove = np.asarray(glove_dict.get(vocab, np.zeros(NCOLS)))
                x[vocab_txt_to_id.get(vocab)] = np.append(wv, glove)
        except IndexError as e:
            print(e)
    x = x.astype('float32')
    return torch.tensor(x)


class SimpleCNN(nn.Module):

    def __init__(self, max_rlen, vocab_size):
        super(SimpleCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, NCOLS * NCOLS_MUL)
        self.conv1 = nn.Conv2d(1, NUM_FILTERS, kernel_size=(KERNEL_SIZE[0], KERNEL_SIZE[1]), stride=1, padding=0)
        # self.batchNorm2d = nn.BatchNorm2d(NUM_FILTERS)
        self.pool = nn.MaxPool2d(kernel_size=(max_rlen - KERNEL_SIZE[0] + 1, 1), stride=1, padding=0)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(NUM_FILTERS * (NCOLS * NCOLS_MUL - KERNEL_SIZE[1] + 1), max_rlen)
        self.fc2 = nn.Linear(max_rlen, NUM_CLASSES)

    def forward(self, x):
        # Computes the activation of the first convolution
        x = self.embedding(x)
        x = self.conv1(x)
        x = torch.relu(x)
        # x = self.batchNorm2d(x)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = torch.relu(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


def main():
    stop_words = utils.get_stopwords()
    vw_model = utils.get_word2vec_model(PATH + 'word2vec/all_word2vector', NCOLS, NWIN)
    vw_model.vectors = utils.normalize2(vw_model.vectors)
    glove_dict = utils.get_glove_data(PATH + 'glove/', 'vectors' + '.txt')
    vocab_txt_to_id, max_rlen = utils.to_number(PATH + "all.txt")
    # vocab_id_to_txt = {v: k for k, v in vocab_txt_to_id.items()}
    vocabs = list(vocab_txt_to_id.keys())
    pretrained_embeddings = get_pretrained(vocab_txt_to_id, vw_model, vocabs, glove_dict)

    # Semeval 2016
    train_list = utils.get_shuffle_list_neutral2(PATH + 'train-pos.txt',
                                                 PATH + 'train-neg.txt', True, stop_words)

    test_list = utils.get_shuffle_list_neutral2(PATH + 'test-pos.txt',
                                                PATH + 'test-neg.txt', False, stop_words)
    # max_rlen = utils.get_max_number_of_token_list(train_list, test_list)
    print("Max Len = " + str(max_rlen))
    print("Data-size = " + str(DATA_SIZE))
    accuracies = []
    for trial in range(NUM_TRIALS):
        model = SimpleCNN(max_rlen, len(vocabs))
        model.embedding.weight.data.copy_(pretrained_embeddings)
        model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        learning_rate = 0.2
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
        y_true = []
        y_pred = []
        y_prob = []
        k_limit = len(train_list) // DATA_SIZE
        for epoch in range(NUM_EPOCHS):
            for k in range(k_limit + 1):
                tt0_batch = time.time()
                train_x, train_y = get_review_windows(vocab_txt_to_id, train_list, max_rlen, k, k_limit)
                train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
                train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)

                for i, (x, labels) in enumerate(train_loader):
                    # Clear gradients w.r.t. parameters
                    optimizer.zero_grad()
                    # Forward pass to get output/logits
                    x.unsqueeze_(1)
                    outputs = model(x.to(device).long())
                    loss = criterion(outputs, labels.to(device=device, dtype=torch.float32))
                    loss.backward()
                    optimizer.step()

                time_elapsed = time.time() - tt0_batch
                print('Trial[{}], Epoch[{}/{}], k[{}/{}], Loss: {:.4f}, Time: {:5.2f}'.
                      format(trial + 1, epoch + 1, NUM_EPOCHS, k + 1, k_limit + 1, loss.item(), time_elapsed))
        print("Path = " + PATH)
        model.eval()
        k_limit = len(test_list) // DATA_SIZE
        for k in range(k_limit + 1):
            # PREDICTIONS
            test_x, test_y = get_review_windows(vocab_txt_to_id, test_list, max_rlen, k, k_limit)
            test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
            test_loader = DataLoader(test_data, shuffle=False, batch_size=1)
            for x, labels in test_loader:
                x.unsqueeze_(1)
                with torch.no_grad():
                    outputs = model(x.to(device).long())
                probabilities = torch.sigmoid(outputs).detach().cpu().numpy().reshape(-1).tolist()
                predicted = []
                for p in probabilities:
                    predicted.append(1 if p > 0.5 else 0)
                labels = labels.cpu().numpy().reshape(-1).tolist()
                y_true.extend(labels)
                y_pred.extend(predicted)
                y_prob.append(probabilities)

        accuracies.append(accuracy_score(y_true, y_pred))
        with open(PATH + "CNN_DYN_scores_" + str(trial + 6 - NUM_TRIALS) + ".txt", 'w') as outfile:
            for t, p, prob in zip(y_true, y_pred, y_prob):
                outfile.write(str(int(t)) + ";" + str(int(p)) + ";" + ";".join(str(p) for p in prob) + "\n")

    print("### ACCURACIES ###")
    for accuracy in accuracies:
        print(f'{accuracy:.8f}')


if __name__ == "__main__":
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")
    main()
