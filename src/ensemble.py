from sklearn.metrics import accuracy_score
from src import constants as con

NUM_TRIALS = 1
path_dic = con.get_paths(con.IMDB)
PATH = path_dic.get(con.WORD)
files_dic = {con.IMDB: [path_dic.get(con.WORD) + "BERT_NN_scores_",
                        path_dic.get(con.WORD) + "BERT_CNN_scores_",
                        path_dic.get(con.WORDPIECE) + "BERT_NN_scores_",
                        path_dic.get(con.WORDPIECE) + "BERT_CNN_scores_",
                        path_dic.get(con.PRE) + "CNN_DYN_scores_",
                        path_dic.get(con.STEM) + "CNN_DYN_scores_",
                        path_dic.get(con.LEMMA) + "CNN_DYN_scores_",
                        path_dic.get(con.PRE) + "RoBERTa_NN_scores_",
                        path_dic.get(con.PRE) + "RoBERTa_CNN_scores_",
                        path_dic.get(con.WORD) + "RoBERTa_NN_scores_",
                        path_dic.get(con.WORD) + "RoBERTa_CNN_scores_"],
             con.SEMEVAL: [path_dic.get(con.WORD) + "BERT_NN_scores_",
                           path_dic.get(con.WORD) + "BERT_CNN_scores_",
                           path_dic.get(con.WORDPIECE) + "BERT_NN_scores_",
                           path_dic.get(con.WORDPIECE) + "BERT_CNN_scores_",
                           path_dic.get(con.STEM) + "BERT_NN_scores_",
                           path_dic.get(con.STEM) + "BERT_CNN_scores_",
                           path_dic.get(con.PRE) + "Dynamic_CNN_scores_",
                           path_dic.get(con.STEM) + "Dynamic_CNN_scores_",
                           path_dic.get(con.WORDPIECE) + "Dynamic_CNN_scores_",
                           path_dic.get(con.WORD) + "RoBERTa_NN_scores_",
                           path_dic.get(con.WORD) + "RoBERTa_CNN_scores_"],
             con.TWITTER: [path_dic.get(con.WORD) + "BERT_NN_scores_",
                           path_dic.get(con.WORD) + "BERT_CNN_scores_",
                           path_dic.get(con.PRE) + "BERT_NN_scores_",
                           path_dic.get(con.PRE) + "BERT_NN_scores_",
                           path_dic.get(con.WORD) + "RoBERTa_NN_scores_",
                           path_dic.get(con.WORD) + "RoBERTa_CNN_scores_",
                           path_dic.get(con.WORDPIECE) + "RoBERTa_NN_scores_",
                           path_dic.get(con.WORDPIECE) + "RoBERTa_CNN_scores_",
                           path_dic.get(con.STEM) + "RoBERTa_NN_scores_",
                           path_dic.get(con.STEM) + "RoBERTa_CNN_scores_"],
             con.BEYAZPERDE: [path_dic.get(con.WORD) + "BERT_NN_scores_",
                              path_dic.get(con.WORD) + "BERT_CNN_scores_",
                              path_dic.get(con.WORDPIECE) + "BERT_NN_scores_",
                              path_dic.get(con.WORDPIECE) + "BERT_NN_scores_",
                              path_dic.get(con.PRE) + "CNN_DYN_scores_",
                              path_dic.get(con.WORD) + "CNN_DYN_scores_",
                              path_dic.get(con.STOPWORDS) + "CNN_DYN_scores_"]
             }


def getPred(filename):
    y_pred = []
    y_true = []
    reverse = False
    with open(filename) as infile:
        for i,line in enumerate(infile):
            score = line.split(";")
            if i == 0 and int(score[0]) == 1:
                reverse = True
            if reverse:
                yp = 1 if int(score[1]) == 0 else 0
                yt = 1 if int(score[0]) == 0 else 0
                y_pred.append(yp)
                y_true.append(yt)
            else:
                y_pred.append(int(score[1]))
                y_true.append(int(score[0]))
    return y_pred, y_true


def ensemble(y_pred_list):
    result = []
    num_of_ensemble = len(y_pred_list)
    length = len(y_pred_list[0])
    for i in range(length):
        preds = []
        for j in range(num_of_ensemble):
            preds.append(y_pred_list[j][i])
        result.append(max(preds, key=preds.count))
    return result


def getFeatures(files, pre=""):
    filenames = []
    for trial in range(NUM_TRIALS):
        for file in files:
            if file.find('BERT') > -1:
                filenames.append(file + str(trial + 1) + pre + ".txt")
            else:
                filenames.append(file + str(trial + 1) + pre + ".txt")
        y_pred_list = []
        for filename in filenames:
            y_pred, y_true = getPred(filename)
            y_pred_list.append(y_pred)
    return y_pred_list, y_true


if __name__ == "__main__":
    for key, value in files_dic.items():
        y_test_list, y_test_true = getFeatures(value)
        print(key + " Max Voting: ")
        results = ensemble(y_test_list)
        accuracy = accuracy_score(y_test_true, results)
        print(f'{accuracy:.8f}')
