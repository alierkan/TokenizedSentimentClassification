import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import utils

BATCH_SIZE = 16
NCOLS = 300
NCOLS_MUL = 2
HIDDEN_DIM = 600
NWIN = 5
NUM_EPOCHS = 25
NUM_CLASSES = 3
NUM_FILTERS = 600
NUM_TRIALS = 10
KERNEL_SIZE = (4, 600)
PATH = "/../"
CUDA = "cuda"
sigmoid = nn.Sigmoid()


def get_review_windows(model, reviews, max_rlen, nsen, glove_dict):
    x = np.zeros(shape=(nsen, max_rlen, NCOLS_MUL * NCOLS))
    y = np.zeros(shape=(nsen, NUM_CLASSES))
    for i, review in enumerate(reviews):
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
            nn.Conv2d(1, NUM_FILTERS, kernel_size=(KERNEL_SIZE[0], KERNEL_SIZE[1]), stride=1, padding=0),
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


def run(train_loader, test_loader, max_len, device):
    accuracies = []
    for trial in range(NUM_TRIALS):
        model = SimpleCNN(max_len)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        learning_rate = 0.1
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)

        total_step = len(train_loader)
        for epoch in range(NUM_EPOCHS):
            for i, (x, labels) in enumerate(train_loader):
                # Clear gradients w.r.t. parameters
                optimizer.zero_grad()
                # Forward pass to get output/logits
                x.unsqueeze_(1)
                outputs = model(x.to(device).float())
                labels = torch.argmax(labels, dim=1)
                loss = criterion(outputs, labels.to(device=device, dtype=torch.int64))
                loss.backward()
                optimizer.step()
                if (i + 1) % 10 == 0:
                    print('Trial[{}], Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.
                          format(trial + 1, epoch + 1, NUM_EPOCHS, i + 1, total_step, loss.item()))

        # PREDICTIONS
        model.eval()
        y_true = []
        y_pred = []
        y_prob = []
        for x, labels in test_loader:
            x.unsqueeze_(1)
            outputs = model(x.to(device).float())
            probabilities = torch.sigmoid(outputs).detach().cpu().numpy().reshape(-1).tolist()
            predicted = [i for i, x in enumerate(probabilities) if x == max(probabilities)]
            labels = labels.cpu().numpy().reshape(-1).tolist()
            observed = [i for i, x in enumerate(labels) if x == max(labels)]
            y_true.extend(observed)
            y_pred.extend(predicted)
            y_prob.append(probabilities)

        accuracies.append(accuracy_score(y_true, y_pred))
        with open(PATH + "CNN3_scores_" + str(trial + 1) + ".txt", 'w') as outfile:
            for t, p, prob in zip(y_true, y_pred, y_prob):
                outfile.write(str(int(t)) + ";" + str(int(p)) + ";" + ";".join(str(p) for p in prob) + "\n")

    print("### ACCURACIES ###")
    for accuracy in accuracies:
        print(f'{accuracy:.8f}')


def main():
    stop_words = utils.get_stopwords()
    vw_model = utils.get_word2vec_model(PATH + 'word2vec/yelp-all_word2vector', NCOLS, NWIN)
    vw_model.vectors = utils.unit(vw_model.vectors)
    glove_dict = utils.get_glove_data(PATH + 'glove/', 'vectors' + '.txt')
    glove_dict = utils.unitDic(glove_dict)

    # Semeval 2016
    train_list = utils.get_shuffle_list_neutral(PATH + 'train-pos.txt',
                                                PATH + 'train-neg.txt',
                                                PATH + 'train-neu.txt', True, stop_words)

    test_list = utils.get_shuffle_list_neutral(PATH + 'test-pos.txt',
                                               PATH + 'test-neg.txt',
                                               PATH + 'test-neu.txt', False, stop_words)

    max_rlen = utils.get_max_number_of_token_list(train_list, test_list)
    train_x, train_y = get_review_windows(vw_model, train_list, max_rlen, len(train_list), glove_dict)
    test_x, test_y = get_review_windows(vw_model, test_list, max_rlen, len(test_list), glove_dict)
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    test_data = TensorDataset(torch.from_numpy(test_x).to(device), torch.from_numpy(test_y).to(device))
    test_loader = DataLoader(test_data, shuffle=False, batch_size=1)
    run(train_loader, test_loader, max_rlen, device)


if __name__ == "__main__":
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device(CUDA)
    else:
        device = torch.device("cpu")
    main()
