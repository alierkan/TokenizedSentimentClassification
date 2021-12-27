import re
import numpy as np
import torch
from transformers import BertTokenizer
import random
import torch.nn as nn
from transformers import BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import time
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_curve, auc

PATH = "/../"
NCOLS = 768
NWIN = 5
NUM_EPOCHS = 10
NUM_CLASSES = 3
NUM_FILTERS = 1024
NUM_TRIALS = 1
KERNEL_SIZE = (4, 768)
MAX_LEN = 768


class MyReviews(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        print(self.filename)
        for line in open(self.filename, encoding="utf8"):
            yield line


def get_shuffle_list_neutral(file_pos, file_neg, file_neu, shuffle):
    xylist = []
    list_pos = MyReviews(file_pos)
    list_neg = MyReviews(file_neg)
    list_neu = MyReviews(file_neu)
    for t in list_pos:
        xylist.append((t, [1, 0, 0]))
    for t in list_neg:
        xylist.append((t, [0, 0, 1]))
    for t in list_neu:
        xylist.append((t, [0, 1, 0]))
    if shuffle:
        random.shuffle(xylist)
    return xylist


def get_review_windows(reviews):
    x = []
    y = []
    for i, review in enumerate(reviews):
        x.append(review[0])
        y.append(review[1])
    return x, y


def text_preprocessing_bert(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)


# Create a function to tokenize a set of texts
def preprocessing_for_bert(data, max_len):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing_bert(sent),  # Preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            truncation=True,
            max_length=max_len,  # Max length to truncate/pad
            # pad_to_max_length=True,  # Pad sentence to max length
            padding='max_length',
            # return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True  # Return attention mask
        )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """

    def __init__(self, max_rlen, kernel_size, class_num, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-large-uncased')

        self.kernel_size = kernel_size
        self.max_rlen = max_rlen
        self.conv1 = nn.Conv2d(1, NUM_FILTERS, kernel_size=(kernel_size[0], kernel_size[1]), stride=1, padding=0)
        self.batchNorm2d = nn.BatchNorm2d(NUM_FILTERS)
        self.pool = nn.MaxPool2d(kernel_size=(max_rlen - kernel_size[0] + 1, 1), stride=1, padding=0)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(NUM_FILTERS * (NCOLS - kernel_size[1] + 1), max_rlen)
        self.fc2 = nn.Linear(max_rlen, class_num)

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        x = outputs[0]
        x.unsqueeze_(1)
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.batchNorm2d(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(-1, NUM_FILTERS * (NCOLS - self.kernel_size[1] + 1))
        x = self.fc1(x)
        x = torch.tanh(x)
        logits = self.fc2(x)

        return logits


def initialize_model(train_dataloader, max_len, epochs=4):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(max_len, KERNEL_SIZE, NUM_CLASSES, freeze_bert=False)

    # Tell PyTorch to run the model on GPU
    bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=5e-5,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler


# Specify loss function
loss_fn = nn.CrossEntropyLoss()


def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def train(model, optimizer, scheduler, train_dataloader, test_dataloader=None, epochs=4, evaluation=False):
    """Train the BertClassifier model.
    """
    # Start training loop
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()
            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            # loss = loss_fn(logits, b_labels)
            loss = loss_fn(logits, torch.max(b_labels, 1)[1])
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(
                    f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

            # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-" * 70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate(model, test_dataloader)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch

            print(
                f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-" * 70)
        print("\n")

    print("Training complete!")


def evaluate(model, test_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = loss_fn(logits, torch.max(b_labels, 1)[1])
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == torch.max(b_labels, 1)[1]).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy


def bert_predict(model, test_dataloader):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)

    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs


def main():
    # Semeval 2016
    train_list = get_shuffle_list_neutral(PATH + 'train-pos.txt',
                                          PATH + 'train-neg.txt',
                                          PATH + 'train-neu.txt', True)
    # trail_list = get_shuffle_list_neutral(PATH + 'trial-pos.txt',
    #                                       PATH + 'trial-neg.txt',
    #                                       PATH + 'trial-neu.txt', True)
    # train_list.extend(trail_list)
    test_list = get_shuffle_list_neutral(PATH + 'test-pos.txt',
                                         PATH + 'test-neg.txt',
                                         PATH + 'test-neu.txt', False)

    train_x, train_y = get_review_windows(train_list)
    test_x, test_y = get_review_windows(test_list)

    # Concatenate train data and test data
    all_text = np.concatenate([train_x, test_x])

    acc_list = []
    for trial in range(NUM_TRIALS):
        # Encode our concatenated data
        encoded_text = [tokenizer.encode(sent, add_special_tokens=True) for sent in all_text]

        # Find the maximum length
        max_len = max([len(sent) for sent in encoded_text])
        print('Max length: ', max_len)

        # Print sentence 0 and its encoded token ids
        token_ids = list(preprocessing_for_bert([train_x[0]], max_len)[0].squeeze().numpy())
        print('Original: ', train_x[0])
        print('Token IDs: ', token_ids)

        # Run function `preprocessing_for_bert` on the train set and the validation set
        print('Tokenizing data...')
        train_inputs, train_masks = preprocessing_for_bert(train_x, max_len)
        test_inputs, test_masks = preprocessing_for_bert(test_x, max_len)

        # Convert other data types to torch.Tensor
        train_labels = torch.tensor(train_y)
        test_labels = torch.tensor(test_y)

        # For fine-tuning BERT, the authors recommend a batch size of 16 or 32.
        batch_size = 16

        # Create the DataLoader for our training set
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        # Create the DataLoader for our validation set
        test_data = TensorDataset(test_inputs, test_masks, test_labels)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

        set_seed(42)  # Set seed for reproducibility
        bert_classifier, optimizer, scheduler = initialize_model(train_dataloader, max_len, epochs=20)
        train(bert_classifier, optimizer, scheduler, train_dataloader, test_dataloader, epochs=20, evaluation=True)

        # Compute predicted probabilities on the test set
        # Please initialize function `bert_predict` by running the first cell in Section 4.2.
        probs = bert_predict(bert_classifier, test_dataloader)

        # Evaluate the Bert classifier
        y_true = np.argmax(np.array(test_y), axis=1)
        y_pred = np.argmax(probs, axis=1)
        acc_list.append(accuracy_score(y_true, y_pred))
        print('################# Trial[{}] ###############'.format(trial + 1))
        with open(PATH + "BERT_CNN_LARGE_scores_" + str(trial+1) + ".txt", 'w') as outfile:
            for t, p, prob in zip(y_true, y_pred, probs.tolist()):
                outfile.write(str(int(t)) + ";" + str(int(p)) + ";" + ";".join(str(pr) for pr in prob) + "\n")
    for accuracy in acc_list:
        print(f'{accuracy:.8f}')


if __name__ == "__main__":
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    main()
