import random
import pandas as pd

def loading(data_name, normalize):
    sentiment_map = {'High': 2, 'Medium': 1, 'Low': 0}
    data = pd.read_csv(data_name, delimiter='\t')
    if normalize:
        if data_name.startswith('test'):
            text = data.text
            sentiments = data.sentiment
        else:
            text, sentiments = equalizer(data, data_name)
    else:
        text = data.text
        sentiments = data.sentiment

    zipped_file = list(zip(text,sentiments))
    random.shuffle(zipped_file)
    text, sentiments = zip(*zipped_file)
    sentiments_indice = [sentiment_map[x] for x in sentiments]
    return text, sentiments_indice

def equalizer(dataset,data_name):
    text = []
    sentiments = []
    sentiment_count = {'High': 0, 'Medium': 0, 'Low': 0}
    if data_name.startswith('train'):
        for row in dataset.iloc:
            sentiment_count[row.sentiment]+=1
            if sentiment_count[row.sentiment]<40000:
                text.append(row.text)
                sentiments.append(row.sentiment)
    elif data_name.startswith('dev'):
        for row in dataset.iloc:
            sentiment_count[row.sentiment]+=1
            if sentiment_count[row.sentiment]<4000:
                text.append(row.text)
                sentiments.append(row.sentiment)
    return text,sentiments



#Train:  {'High': 87638, 'Medium': 21402, 'Low': 36127}
#Test:  {'High': 32349, 'Medium': 7920, 'Low': 13497}
#Val: {'High': 9772, 'Medium': 2410, 'Low': 3948}
#
# if __name__ == '__main__':
#     loading('dev.tsv')