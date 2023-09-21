import torch.optim
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import dataloader
from Model import LLM_LSTM, LLM_CNN
from transformers import BertTokenizer, BertModel, RobertaModel,DistilBertTokenizer, DistilBertModel, RobertaTokenizer
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
import argparse
import transformers
import random
import time
import torch


transformers.logging.set_verbosity_error()
parser = argparse.ArgumentParser(description="Train a sequence classifier - via Transformers")
parser.add_argument("--BERT", type=bool, help="You are using BERT embeddings", default=False)
parser.add_argument("--RoBERTa", type=bool, help="You are using RoBERTa embeddings", default=False)
parser.add_argument("--DistilBERT", type=bool, help="You are using DistilBERTa embeddings", default=True)

parser.add_argument("--train_file", type=str, help="train dataset name", default='train.tsv')
parser.add_argument("--test_file", type=str, help="test dataset name", default='test.tsv')
parser.add_argument("--eval_file", type=str, help="eval dataset name", default='dev.tsv')
parser.add_argument("--normalized", type=bool, help="normalize the dataset labels frequency", default=True)

parser.add_argument("--lstm", type=bool, help="You are using LSTMs", default=False)
parser.add_argument("--cnn", type=bool, help="You are using CNN", default=True)

parser.add_argument("--epoch", type=int, help="this is the number of epochs", default=2)
parser.add_argument("--hiddens", type=int, help="this is the LSTM hidden_size", default=100)
parser.add_argument("--batch_size", type=int, help="number of samples in each iteration", default=10)
parser.add_argument("--lr", type=float, help="this is learning rate value", default=0.005)
parser.add_argument("--num_labels", type=int, help="this is the total number of labels", default=3)
parser.add_argument("--max_length", type=int, help="this is maximum length of an utterance", default=200)
parser.add_argument("--filter_widths", type=list, help="list of integers defines the width of filters", default=[2,3])
parser.add_argument("--num_conv_layers", type=int, help="defines number of the convolutional layers", default=2)
parser.add_argument("--num_filters", type=int, help="defines number of filters in CNN", default=2)
parser.add_argument("--intermediate_pool_size", type=int, help="defines intermediate_pool_size for max pool", default=3)

parser.add_argument("--L1_reg", type=bool, help="L1 regularizer", default=False)
parser.add_argument("--L2_reg", type=bool, help="L2 regularizer", default=False)
parser.add_argument("--drop_out", type=bool, help="implement a dropout to the model output", default=True)
parser.add_argument("--L1_lambda", type=int, help="Lambda value used for regularization", default=0.01)
parser.add_argument("--L2_lambda", type=int, help="Lambda value used for regularization", default=0.1)
parser.add_argument("--dr", type=int, help="P value used for Dropout", default=0.3)
args = parser.parse_args()

if args.BERT:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encoder_model = BertModel.from_pretrained("bert-base-uncased", num_labels=args.num_labels)
elif args.RoBERTa:
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', cache_dir='RoBERT_CacheDir')
    encoder_model = RobertaModel.from_pretrained('roberta-base', num_labels=args.num_labels)
elif args.DistilBERT:
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    encoder_model = DistilBertModel.from_pretrained('distilbert-base-uncased', num_labels=args.num_labels)


if args.lstm:
    embedding_dim = encoder_model.config.hidden_size
    model_object = LLM_LSTM(embedding_dim,args.num_labels, args.hiddens, args.dr)
else:
    model_object = LLM_CNN(args.filter_widths, args.num_conv_layers, args.num_filters, args.num_labels, args.dr, args.max_length, args.intermediate_pool_size)

loss = CrossEntropyLoss()
if args.L2_reg:
    optimizer = torch.optim.Adam(model_object.parameters(), lr=args.lr, weight_decay= args.L2_lambda)
else:
    optimizer = torch.optim.Adam(model_object.parameters(), lr=args.lr)

loss_records = []
def train_classifier():
    text, label = dataloader.loading('train.tsv',True)
    text = text[:50]
    label = label[:50]
    patience = 0
    curr_loss = 0
    for epoch_indx in range(args.epoch):
        prev_loss = curr_loss
        epoch_loss = 0
        acc_epoch_record = []
        loss_epoch_record = []
        text,label = randomize(text,label)
        for batch_indx in tqdm(range(0, len(text), args.batch_size), desc= f"TRAINING DATASET: {epoch_indx+1}/{args.epoch}"):
            batch_encoding = tokenizer.batch_encode_plus(
                text[batch_indx:batch_indx + args.batch_size], padding='max_length', truncation=True,
                max_length=args.max_length, return_tensors='pt', text_pair=True, token_type_ids=True)
            input_ids = batch_encoding['input_ids']
            attention_mask = batch_encoding['attention_mask']
            out = encoder_model(input_ids)
            embeddings = out.last_hidden_state
            predicted = model_object.forward(embeddings, attention_mask)
            gold_label_tensor = torch.tensor(label[batch_indx:batch_indx + args.batch_size])
            preds = torch.argmax(predicted, dim=1)
            accuracy = calculate_accuracy(gold_label_tensor, preds)
            acc_epoch_record.append(accuracy)
            loss_value = loss(predicted, gold_label_tensor)
            if args.L1_reg:
                for param in model_object.parameters():
                    loss_value += torch.sum(torch.abs(param)) * args.L1_lambda
            loss_epoch_record.append(loss_value.item())
            epoch_loss += loss_value.item()
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Ave TRAIN acc: Epoch {epoch_indx + 1}: {accuracy}")
        print(f"Ave TRAIN loss: Epoch {epoch_indx + 1}: {sum(loss_epoch_record)/len(loss_epoch_record)}")
        loss_records.append(epoch_loss)
        eval_txt, eval_lbl = dataloader.loading('dev.tsv', True)
        eval_txt = eval_txt[:80]
        eval_lbl = eval_lbl[:80]
        eval_txt, eval_lbl = randomize(eval_txt, eval_lbl)
        acc,ave_loss,_,_ = evaluation(model_object, eval_txt, eval_lbl)
        print(f"Ave DEV acc: Epoch {epoch_indx+1}: {acc}")
        print(f"Ave DEV loss: Epoch {epoch_indx + 1}: {ave_loss}")
        curr_loss = ave_loss
        if curr_loss >= prev_loss: # Implementing Early Stopping
            patience+=1
            if patience>2:
                test_txt, test_lbl = dataloader.loading('test.tsv',True)
                test_txt = test_txt[:80]
                test_lbl = test_lbl[:80]
                acc, ave_loss, total_predicted_label, total_gold_label = evaluation(model_object, test_txt, test_lbl)
                print(f"Ave TEST acc: {acc}")
                print(f"Ave TEST loss: Epoch {epoch_indx + 1}: {ave_loss}")
                matrices(total_gold_label, total_predicted_label)
                return
        else:
            patience=0
    test_txt, test_lbl = dataloader.loading('test.tsv',True)
    test_txt = test_txt[:80]
    test_lbl = test_lbl[:80]
    acc, ave_loss,  total_predicted_label, total_gold_label = evaluation(model_object, test_txt, test_lbl )
    print(f"Ave TEST acc: {acc}")
    print(f"Ave TEST loss: Epoch {epoch_indx + 1}: {ave_loss}")
    matrices(total_gold_label, total_predicted_label)


def evaluation(model, text, label):
    ave_loss_epoch = []
    total_predicted_label = []
    total_gold_label = []
    with torch.no_grad():
        for batch_indx in range(0, len(label), args.batch_size):
            batch_encoding = tokenizer.batch_encode_plus(
                text[batch_indx:batch_indx + args.batch_size], padding='max_length',max_length=args.max_length, truncation=True, return_tensors='pt', text_pair=True)
            input_ids = batch_encoding['input_ids']
            attention_mask = batch_encoding['attention_mask']
            out = encoder_model(input_ids)
            embeddings = out.last_hidden_state
            predicted = model.forward(embeddings, attention_mask)
            gold_label_list = label[batch_indx:batch_indx + args.batch_size]
            total_gold_label.extend(gold_label_list)
            gold_label_tensor = torch.tensor(gold_label_list)
            loss_value = loss(predicted, gold_label_tensor)
            ave_loss_epoch.append(loss_value)
            preds = torch.argmax(predicted, dim=1)
            total_predicted_label.extend(preds)
            accuracy = calculate_accuracy(gold_label_tensor, preds)
        return accuracy, sum(ave_loss_epoch)/len(ave_loss_epoch), total_predicted_label, total_gold_label

def randomize(list1, list2):
    combined = list(zip(list1, list2))
    random.shuffle(combined)
    shuffled_list1, shuffled_list2 = zip(*combined)
    return shuffled_list1,shuffled_list2

def plotting(records):
    batchList = [i for i in range(len(records))]
    plt.plot(batchList, records, linewidth=5, label="Loss variation")
    plt.xlabel("Batch", color="green", size=20)
    plt.ylabel("Loss", color="green", size=20)
    plt.title("Progress Line for the Model", size=20)
    plt.grid()
    plt.show()

def calculate_accuracy(gold_label, predicted):
    correct = torch.sum(gold_label == predicted).item()
    total = len(gold_label)
    accuracy = (correct / total) * 100
    return accuracy

def matrices(gold, predicted):
    predicted = [tensor.cpu() for tensor in predicted]
    results = classification_report(gold, predicted)
    print(results)
    print()
    cm = confusion_matrix(gold, predicted)
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()
    
if __name__ == '__main__':
    start_time = time.time()
    train_classifier()
    plotting(loss_records)
    seconds = time.time() - start_time
    print('Time Taken:', time.strftime("%H:%M:%S", time.gmtime(seconds)))