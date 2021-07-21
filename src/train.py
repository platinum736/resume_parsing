import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from src.model import Ner
import math
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from src.create_train_data import *
import nltk
import os
import spacy
import sys
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    os.system('python -m spacy download en_core_web_sm')
    print('need to download spacy model')
    sys.exit()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
dirname = os.path.dirname(__file__)

vocab = {}
data_dir = os.path.join(dirname, '../data/')
model_dir = os.path.join(dirname, '../model/')
word2id = dict()
tag_map = {}
id2tag = dict()


def loss_fn(outputs, labels):
    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.view(-1)

    # mask out 'PAD' tokens
    mask = (labels >= 0).float()

    # the number of tokens is the sum of elements in mask
    num_tokens = int(torch.sum(mask).data.item())

    # pick the values corresponding to labels and multiply by mask
    outputs = outputs[range(outputs.shape[0]), labels]*mask

    # cross entropy loss for all non 'PAD' tokens
    return -torch.sum(outputs)/num_tokens


def get_vocab():
    with open(data_dir+'vocab.txt') as f:
        for i, l in enumerate(f.read().splitlines()):
            vocab[l] = i
    return vocab


def word_id(vocab):
    word2id = dict()
    for word in vocab:
        ix = vocab[word]
        word2id[ix] = word
    return word2id


def get_tags():
    with open(data_dir+'tags.txt') as f:
        for i, l in enumerate(f.read().splitlines()):
            tag_map[l] = i
    return tag_map


def id_tag(tag_map):
    id2tag = dict()
    for tag in tag_map:
        ix = tag_map[tag]
        id2tag[ix] = tag
    return id2tag


def create_batch(train_data, train_labels, i, batch_size=32):
    # compute length of longest sentence in batch
    st = i*batch_size
    en = st+batch_size
    batch_sentences = train_data[st:en]
    batch_tags = train_labels[st:en]

    batch_max_len = max([len(s) for s in batch_sentences])

    # prepare a numpy array with the data, initializing the data with 'PAD'
    # and all labels with -1; initializing labels to -1 differentiates tokens
    # with tags from 'PAD' tokens
    batch_data = vocab['PAD']*np.ones((len(batch_sentences), batch_max_len))
    batch_labels = -1*np.ones((len(batch_sentences), batch_max_len))

    # copy the data to the numpy array
    for j in range(len(batch_sentences)):
        cur_len = len(batch_sentences[j])
        batch_data[j][:cur_len] = batch_sentences[j]
        batch_labels[j][:cur_len] = batch_tags[j]

    # since all data are indices, we convert them to torch LongTensors
    batch_data, batch_labels = torch.LongTensor(
        batch_data), torch.LongTensor(batch_labels)

    # convert Tensors to Variables
    batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)
    return batch_data, batch_labels


def train():

    model = Ner({'vocab_size': len(vocab), 'embedding_dim': 200,
                'lstm_hidden_dim': 100, 'number_of_tags': len(tag_map)})

    num_epochs = 4
    batch_size = 32
    train_sentences = []
    train_labels = []
    train_file = 'train_resume.bie'
    with open(data_dir+train_file) as f:
        s = list()
        t = list()
        for line in f.read().splitlines():
            # replace each token by its index if it is in vocab
            # else use index of UNK
            words = line.split()
            if len(words) < 2:
                train_sentences.append(s)
                train_labels.append(t)
                s = list()
                t = list()
            else:
                word = words[0]
                tag = words[-1]
                if word in vocab:
                    s.append(vocab[word])
                else:
                    s.append(vocab['UNK'])
                if tag in tag_map:
                    t.append(tag_map[tag])
                else:
                    t.append(tag_map['O'])
    num_training_steps = (len(train_sentences)/batch_size)
    i = 0
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(num_epochs):
        for i in range(math.ceil(num_training_steps)):
            print(i, end=' ')
            batch_sentences, batch_labels = create_batch(
                train_sentences, train_labels, i, 32)
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            output_batch = model(batch_sentences)
            # calculate loss
            loss = loss_fn(output_batch, batch_labels)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
            print(loss_fn(output_batch, batch_labels))

        # Specify a path
    PATH = "../model/resume_ner_model.pt"

    # Save
    torch.save(model.state_dict(), PATH)


def create_data(text, vocab):
    test_ids = list()
    s_ids = list()
    for line in text.splitlines():
        # replace each token by its index if it is in vocab
        # else use index of UNK
        words = line.split()
        for word in words:
            if word in vocab:
                s_ids.append(vocab[word])
            else:
                s_ids.append(vocab['UNK'])
        test_ids.append(s_ids)
        s_ids = list()
    return test_ids


def create_test_batch(test_data, i, batch_size=32):
    # compute length of longest sentence in batch
    st = i*batch_size
    en = st+batch_size
    batch_sentences = test_data[st:en]

    batch_max_len = max([len(s) for s in batch_sentences])

    # prepare a numpy array with the data, initializing the data with 'PAD'
    # and all labels with -1; initializing labels to -1 differentiates tokens
    # with tags from 'PAD' tokens
    batch_data = vocab['PAD']*np.ones((len(batch_sentences), batch_max_len))

    # copy the data to the numpy array
    for j in range(len(batch_sentences)):
        cur_len = len(batch_sentences[j])
        batch_data[j][:cur_len] = batch_sentences[j]

    # since all data are indices, we convert them to torch LongTensors
    batch_data = torch.LongTensor(batch_data)

    # convert Tensors to Variables
    batch_data = Variable(batch_data)
    return batch_data

# mostly first line would be name if not enter manually


def extract_email(text):
    email_regex = re.compile(
        r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.])")
    res = re.search(email_regex, text)
    if res:
        span = res.span()
        return text[span[0]:span[1]]
    else:
        return None


def extract_phone(text):
    phone_regex = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
    res = re.search(phone_regex, text)
    if res:
        span = res.span()
        return text[span[0]:span[1]]
    else:
        return None


def extract_Name(text):
    name = nltk.tokenize.sent_tokenize(text.strip())[0].split('\n')[0]
    return name


def extract_exp(text):
    doc = nlp(text)
    entities = dict()
    for _ in doc.ents:
        label = _.label_.lower()
        entity = _.text.lower()
        if label not in entities:
            entities[label] = list()
            entities[label].append(entity)
        else:
            entities[label].append(entity)
    return entities['date'][0]


def predict(file_path):
    vocab = get_vocab()
    word2id = word_id(vocab)

    tag_map = get_tags()
    tag2id = id_tag(tag_map)

    model = Ner({'vocab_size': len(vocab), 'embedding_dim': 200,
                 'lstm_hidden_dim': 100, 'number_of_tags': len(tag_map)})
    model.load_state_dict(torch.load(model_dir + 'resume_ner_model.pt'))

    text = read_file(file_path)
    test_data = create_data(text, vocab)
    batch_size = 32
    num_batches = math.ceil(len(test_data)/batch_size)
    extracted_tags = dict()
    extracted_tags['name'] = extract_Name(text)
    extracted_tags['experience'] = extract_exp(text)
    extracted_tags['email'] = extract_email(text)
    extracted_tags['phone'] = extract_phone(text)
    for batch in range(num_batches):
        batch_sent = create_test_batch(test_data, batch, 32)
        res = model(batch_sent)
        tags = np.argmax(res.detach().numpy(), axis=1)
        shape = batch_sent.shape
        tags = tags.reshape(shape[0], shape[1])
        batch_sent = batch_sent.detach().numpy()
        for row in range(len(batch_sent)):
            for col in range(len(batch_sent[row])):
                if tags[row][col] not in [0, 1, 2]:
                    ix = batch_sent[row][col]
                    ent_type = tag2id[tags[row][col]].split('-')[1]
                    ent = word2id[ix]
                    if ent_type not in extracted_tags:
                        extracted_tags[ent_type] = list()
                    if tag2id[tags[row][col]].split('-')[0] == 'I':
                        extracted_tags[ent_type][-1] += ' '+ent
                    else:
                        extracted_tags[ent_type].append(ent)
    return extracted_tags
