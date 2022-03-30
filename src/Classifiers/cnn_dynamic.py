import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from src import utils
import torch.nn.functional as F

BATCH_SIZE = 16
NCOLS = 300
NCOLS_MUL = 2
NWIN = 5
NUM_EPOCHS = 25
NUM_CLASSES = 3
NUM_FILTERS = 600
NUM_TRIALS = 10
KERNEL_SIZE = (4, 600)
PATH = "/../"  # Has to be defined.
WORD2VEC_PATH = PATH + 'word2vec.txt'
GLOVE_FILE = "glove.txt"


def get_review_windows(vocab_txt_to_id, reviews, max_rlen, nsen):
    x = np.zeros(shape=(nsen, max_rlen))
    y = np.zeros(shape=(nsen, NUM_CLASSES))
    for i, review in enumerate(reviews):
        try:
            x[i] = [vocab_txt_to_id.get(r) for r in review[0]] + (max_rlen - len(review[0]))*[0]
        except IndexError as e:
            print(e)
        y[i] = review[1]
    x.astype('int32')
    y.astype('int')
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
        self.embedding = nn.Embedding(vocab_size, NCOLS * NCOLS_MUL, padding_idx=max_rlen)
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
    vw_model = utils.get_word2vec_model(WORD2VEC_PATH, NCOLS, NWIN)
    vw_model.vectors = utils.normalize2(vw_model.vectors)
    glove_dict = utils.get_glove_data(PATH, GLOVE_FILE)
    vocab_txt_to_id, max_rlen = utils.to_number(PATH + "all.txt")  # Combine all files as all.txt
    vocabs = list(vocab_txt_to_id.keys())
    pretrained_embeddings = get_pretrained(vocab_txt_to_id, vw_model, vocabs, glove_dict)

    if NUM_CLASSES == 3:
        train_list = utils.get_shuffle_list_neutral(PATH + 'train-pos.txt',
                                                    PATH + 'train-neg.txt',
                                                    PATH + 'train-neu.txt', True, stop_words)

        test_list = utils.get_shuffle_list_neutral(PATH + 'test-pos.txt',
                                                   PATH + 'test-neg.txt',
                                                   PATH + 'test-neu.txt', False, stop_words)
    elif NUM_CLASSES == 2:
        train_list = utils.get_shuffle_list(PATH + 'train-pos.txt',
                                            PATH + 'train-neg.txt', True, stop_words)

        test_list = utils.get_shuffle_list(PATH + 'test-pos.txt',
                                           PATH + 'test-neg.txt', False, stop_words)

    train_x, train_y = get_review_windows(vocab_txt_to_id, train_list, max_rlen, len(train_list))
    test_x, test_y = get_review_windows(vocab_txt_to_id, test_list, max_rlen, len(test_list))

    train_data = TensorDataset(torch.from_numpy(train_x).to(device), torch.from_numpy(train_y).to(device))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    test_data = TensorDataset(torch.from_numpy(test_x).to(device), torch.from_numpy(test_y).to(device))
    test_loader = DataLoader(test_data, shuffle=False, batch_size=1)

    accuracy_score_list = []
    for trial in range(NUM_TRIALS):
        model = SimpleCNN(max_rlen, len(vocabs))
        model.embedding.weight.data.copy_(pretrained_embeddings)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        learning_rate = 0.2
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)

        total_step = len(train_loader)
        for epoch in range(NUM_EPOCHS):
            for i, (x, labels) in enumerate(train_loader):
                # Clear gradients w.r.t. parameters
                optimizer.zero_grad()
                # Forward pass to get output/logits
                x.unsqueeze_(1)
                outputs = model(x.to(device).long())
                labels = torch.argmax(labels, dim=1)
                loss = criterion(outputs, labels.to(device=device, dtype=torch.int64))
                loss.backward()
                optimizer.step()
                if (i + 1) % 100 == 0:
                    print('Trial[{}], Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.
                          format(trial + 1, epoch + 1, NUM_EPOCHS, i + 1, total_step, loss.item()))

        model.eval()
        y_true = []
        y_pred = []
        y_probabilities = []

        for x, labels in test_loader:
            x.unsqueeze_(1)
            # Forward pass only to get logits/output
            outputs = model(x.to(device).long())

            # PREDICTIONS
            probabilities = outputs.detach().cpu().numpy().reshape(-1).tolist()
            predicted = [i for i, x in enumerate(probabilities) if x == max(probabilities)]
            labels = labels.cpu().numpy().reshape(-1).tolist()
            observed = [i for i, x in enumerate(labels) if x == max(labels)]
            y_true.extend(observed)
            y_pred.extend(predicted)
            y_probabilities.append(outputs)
        accuracy_score_list.append(accuracy_score(y_true, y_pred))
        y_probabilities = torch.cat(y_probabilities, dim=0)
        y_probabilities = F.softmax(y_probabilities, dim=1).cpu().detach().numpy()
        with open(PATH + "Dynamic_CNN_scores_" + str(trial+101) + ".txt", 'w') as outfile:
            for t, p, prob in zip(y_true, y_pred, y_probabilities):
                prob_str = ",".join([str(p) for p in prob])
                outfile.write(str(int(t)) + ";" + str(int(p)) + ";" + prob_str + "\n")
    print("### ACCURACIES ###")
    for acc in accuracy_score_list:
        print(acc)


if __name__ == "__main__":
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    main()
