from tika import parser
import os
import pandas as pd
from ast import literal_eval
import random
import tqdm
import re
import numpy as np
import tika
tika.initVM()

#files = os.listdir('../data/Resumes')
data_dir = '../data/'
# labelled data for name and city
#resume_entities = pd.read_csv(data_dir+'/resume_entities1.csv')


def clean_text(text):
    text = text.strip()
    text = re.sub(r'\t+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.lower()


def read_file(file_path):
    parsed = parser.from_file(file_path)
    # print(parsed["metadata"])
    # print(parsed["content"])
    text = clean_text(parsed['content'])
    return text


# words followed by 'client:'
def extract_organisation(text):
    lines = text.split('\n')
    orgs = list()
    for line in lines:
        if line.find('client:') > -1:
            if len(line.split(':')) > 1:
                ent = line.split(':')[1].split(',')[0].strip()
                if ent != '':
                    org = nltk.tokenize.sent_tokenize(ent)[0]
                    # print(org)
                    orgs.append(org)
    # print(orgs)
    return orgs

# words followed by environment:


def extract_skills(text):
    skills = list()
    for line in text.split('\n'):
        line = line.lower()
        if line.find('environment:') > -1:
            skills.extend(line.replace('environment:', '').split(','))
    return skills


def get_matches():
    all_file_matches = list()
    files = list(resume_entities['file'])
    for file in tqdm.tqdm(files):
        text = read_file(file)
        all_matches = dict()
        for skill in all_skills:
            skill = skill.strip()
            if skill == '':
                continue
            matches = [_.span() for _ in re.finditer(
                r'\b(%s)\b' % re.escape(skill), text)]
            if len(matches) > 0:
                all_matches[skill] = matches
        all_file_matches.append(all_matches)
    return all_file_matches


def get_entity_indexes(doc):
    text = read_file(doc['file']).lower()
    matches = list()
    print(doc['file'])
    for col in ['skill_matches', 'org_matches', 'experience', 'person name', 'city']:
        entities = doc[col]
        if col == 'skill_matches':
            col = 'skill'
            #entities = literal_eval(entities)
        elif col == 'org_matches':
            col = 'org'
            #entities = literal_eval(entities)
        elif col == 'person name':
            col = 'person'
        if entities == 'None':
            continue
        if type(entities) in [dict, list]:
            for ent in entities:
                matches.extend(
                    [_.span()+(col,) for _ in re.finditer(r'\b(%s)\b' % re.escape(ent.lower().strip()), text)])
        elif type(entities) is str:
            temp = [_.span()+(col,) for _ in re.finditer(r'\b(%s)\b' %
                                                         re.escape(entities.lower().strip()), text)]
            if len(temp) > 0:
                matches.append(temp[0])
        else:
            continue
    return matches


def write_train_data(file, labels, fp):
    newline = ' '
    text = read_file(file)
    #data_tagged = 32
    count = -1
    count += 1
    # if count == data_tagged:
    #    break
    ann_last = 0
    print(labels)
    dtype = [('start', int), ('end', int), ('label', 'U15')]
    values = list()
    for label in labels:
        values.append((label[0], label[1], label[2]))
    temp_list = np.array(values, dtype=dtype)
    temp_list = np.sort(temp_list, order=['start'])
    for ann in temp_list:
        print(ann_last)
        outside_words = text[ann_last:ann[0]].split(' ')
        for word in outside_words:
            if word != '':
                fp.write(word + ' ' + 'O'+'\n')
        inside_words = text[ann[0]:ann[1]].split(' ')
        label = ann[2]
        ix = 0
        for word in inside_words:
            if word != ' ':
                if ix == 0:
                    fp.write(word + ' B-'+label+'\n')
                else:
                    fp.write(word + ' I-'+label+'\n')
                ix += 1
        ann_last = ann[1]
    outside_words = text[ann_last:len(text)].split(' ')
    for word in outside_words:
        if word != '':
            fp.write(word + ' ' + 'O'+'\n')
    fp.write("\n")


def build_vocab():
    vocab = set()
    tags = set()
    with open('train_resume.bie') as fp:
        for line in fp.readlines():
            words = line.split()
            if len(words) == 2:
                word = words[0]
                tag = words[1]
                vocab.add(word)
                tags.add(tag)


def prepare_train_data():
    organisations = list()
    for ix in tqdm.tqdm(resume_entities.index):
        doc = resume_entities.loc[ix]
        text = read_file(doc['file'])
        organisations.append(extract_organisation(text))
    resume_entities['Organisations'] = organisations

    skills = list()
    for ix in tqdm.tqdm(resume_entities.index):
        doc = resume_entities.loc[ix]
        text = read_file(doc['file'])
        skills.append(extract_skills(text))
    resume_entities['skills'] = skills
    all_skills = list()
    #x = [all_skills.extend(literal_eval(_)) for _ in resume_entities['skills']]
    x = [all_skills.extend(_) for _ in resume_entities['skills']]
    all_skills = set(all_skills)
    all_skill_matches = get_matches()
    resume_entities['skill_matches'] = all_skill_matches

    all_entities = list()
    for ix in resume_entities.index:
        doc = resume_entities.loc[ix]
        all_entities.append(get_entity_indexes(doc))
    resume_entities['entities'] = all_entities
    train_data = resume_entities[(resume_entities['city'] != 'None')]

    with open(data_dir+'train_resume.bie', 'w') as fp:
        for ix in train_data.index:
            doc = train_data.loc[ix]
            write_train_data(doc['file'], doc['entities'], fp)

    build_vocab()
    with open(data_dir+'vocab.txt', 'w') as fp:
        for word in vocab:
            fp.writelines(word+'\n')

    with open(data_dir+'tags.txt', 'w') as fp:
        for tag in tags:
            fp.writelines(tag+'\n')
