from src import utils
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src import constants as con
import time

DATA_SIZE = 1024
BATCH_SIZE = 16
NCOLS = 300
NCOLS_MUL = 2
HIDDEN_DIM = 600
NWIN = 5
NUM_EPOCHS = 25
NUM_CLASSES = 1
NUM_FILTERS = 600
NUM_TRIALS = 5
KERNEL_SIZE = (4, 600)
PATH = "/../"
sigmoid = nn.Sigmoid()


def get_vector_weight(model, x):
    xu = x.unsqueeze(1)
    outputs = model(xu.to(device).float())
    outputs.squeeze_()
    outputs = sigmoid(outputs)
    probabilities = outputs.detach().cpu().numpy()
    avg = np.average(probabilities, axis=1)
    polarity = np.zeros(probabilities.shape[0])
    for i, p in enumerate(probabilities):
        scalar = (p - avg[i]) / avg[i] + 1
        polarity[i] = np.dot(scalar, scalar)
    scalar_vector = np.log2(polarity)
    for i, s in enumerate(scalar_vector):
        x[i] = s * x[i]
    return x


def get_review_windows(model, reviews, max_rlen, glove_dict, index, index_limit):
    data_size = DATA_SIZE if index < index_limit else (len(reviews) - DATA_SIZE * index_limit)
    x = np.zeros(shape=(data_size, max_rlen, NCOLS_MUL * NCOLS))
    y = np.zeros(shape=(data_size, NUM_CLASSES))
    start_index = index * DATA_SIZE
    for i, review in enumerate(reviews[start_index: start_index + data_size]):
        try:
            x[i] = utils.get_token_matrix(model, review[0], max_rlen, NCOLS, glove_dict, NCOLS_MUL)
        except IndexError as e:
            print(e)
        y[i] = review[1]
    x = x.astype('float32')
    return x, y


class SimpleCNN(nn.Module):
    def __init__(self, max_rlen):
        super(SimpleCNN, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, NUM_FILTERS, kernel_size=(KERNEL_SIZE[0], KERNEL_SIZE[1]), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(max_rlen - KERNEL_SIZE[0] + 1, 1), stride=1, padding=0))
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(NUM_FILTERS * (HIDDEN_DIM - KERNEL_SIZE[1] + 1), max_rlen)
        self.fc2 = nn.Linear(max_rlen, NUM_CLASSES)

    def forward(self, x):
        x = self.layer(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

def main():
    stop_words = utils.get_stopwords()
    vw_model = utils.get_word2vec_model(PATH + 'word2vec/all_word2vector', NCOLS, NWIN)
    vw_model.vectors = utils.unit(vw_model.vectors)
    glove_dict = utils.get_glove_data(PATH + 'glove/', 'vectors' + '.txt')
    glove_dict = utils.unitDic(glove_dict)

    # IMDB
    train_list = utils.get_shuffle_list_neutral2(PATH + 'train-pos.txt',
                                                 PATH + 'train-neg.txt', True, stop_words)

    test_list = utils.get_shuffle_list_neutral2(PATH + 'test-pos.txt',
                                                PATH + 'test-neg.txt', False, stop_words)

    max_rlen = utils.get_max_number_of_token_list(train_list, test_list)
    print("Max Len = " + str(max_rlen))
    print("Data-size = " + str(DATA_SIZE))
    accuracies = []
    for trial in range(NUM_TRIALS):
        model = SimpleCNN(max_rlen)
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
                train_x, train_y = get_review_windows(vw_model, train_list, max_rlen, glove_dict, k, k_limit)
                train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
                train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)

                for i, (x, labels) in enumerate(train_loader):
                    # Clear gradients w.r.t. parameters
                    optimizer.zero_grad()
                    # Forward pass to get output/logits
                    x.unsqueeze_(1)
                    outputs = model(x.to(device).float())
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
            test_x, test_y = get_review_windows(vw_model, test_list, max_rlen, glove_dict, k, k_limit)
            test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
            test_loader = DataLoader(test_data, shuffle=False, batch_size=1, pin_memory=True)
            for x, labels in test_loader:
                x.unsqueeze_(1)
                with torch.no_grad():
                    outputs = model(x.to(device).float())
                probabilities = torch.sigmoid(outputs).detach().cpu().numpy().reshape(-1).tolist()
                predicted = []
                for p in probabilities:
                    predicted.append(1 if p > 0.5 else 0)
                labels = labels.cpu().numpy().reshape(-1).tolist()
                y_true.extend(labels)
                y_pred.extend(predicted)
                y_prob.append(probabilities)

        accuracies.append(accuracy_score(y_true, y_pred))
        with open(PATH + "CNN_scores_" + str(trial + 6 - NUM_TRIALS) + ".txt", 'w') as outfile:
            for t, p, prob in zip(y_true, y_pred, y_prob):
                outfile.write(str(int(t)) + ";" + str(int(p)) + ";" + ";".join(str(p) for p in prob) + "\n")

    print("### ACCURACIES ###")
    for accuracy in accuracies:
        print(f'{accuracy:.8f}')


if __name__ == "__main__":
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    main()
