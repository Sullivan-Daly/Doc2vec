# gensim modules

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.metrics import jaccard_similarity_score

from math import*

import scipy

# configparser
import configparser
import difflib

# random shuffle
from random import shuffle

# numpy
import numpy

# classifier
from sklearn.linear_model import LogisticRegression

def ConfigReader():
    tConfig = {}
    xConfig = configparser.ConfigParser()
    xConfig.read('config.ini')
    tConfig['fOrianePos'] = xConfig['FILE']['FOrianePos']
    tConfig['fOrianeNeg'] = xConfig['FILE']['FOrianeNeg']
    tConfig['fCorpus2016'] = xConfig['FILE']['FCorpus2016']
    tConfig['sOptionLoad'] = xConfig['OPTION']['Load']
    tConfig['sOptionSave'] = xConfig['OPTION']['Save']
    tConfig['sOptionModelName'] = xConfig['OPTION']['ModelName']
    tConfig['sResultFile'] = xConfig['RESULT']['FileName']
    tConfig['sMabedFile'] = xConfig['MABED']['FileName']
    return tConfig

def ConfigPrint(tConfig):
    print('fOrianePos -> ' + tConfig['fOrianePos'])
    print('fOrianeNeg -> ' + tConfig['fOrianeNeg'])
    print('fCorpus2016 -> ' + tConfig['fCorpus2016'])
    print('sOptionLoad -> ' + tConfig['sOptionLoad'])
    print('sOptionSave -> ' + tConfig['sOptionSave'])

# LabeledSentence or TaggedDocument from gensim.models.doc2vec
class ModelTest(object):

    tConfig = {}

    @staticmethod
    def ModelLoad(tConfig):
        tmpModel = Doc2Vec(min_count=1, window=5, size=100, sample=1e-4, negative=5, workers=7)
        tmpModel.load(tConfig['sOptionModelName'])
        return tmpModel

    def __init__(self, tConfig):
        self.tConfig = tConfig


class TaggedTweet(object):
    def __init__(self, dSources):
        self.dSources = dSources

        dFlipped = {}

        # make sure that keys are unique
        for key, value in dSources.items():
            if value not in dFlipped:
                dFlipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.dSources.items():
            with open(source, "r", encoding='utf16') as file:
                for item_no, row in enumerate(file):
                    yield TaggedDocument(utils.to_unicode(row.rsplit('; ', maxsplit=2)[2]).split(),
                                         [prefix + '_%s' % item_no])


    def to_array(self):
        self.sentences = []
        numbers = [0, 0, 0]
        for source, prefix in self.dSources.items():
            with open(source, "r", encoding='utf16') as file:
                print(source)
                for item_no, row in enumerate(file):
                    if not item_no == 0:
                        sentence = row.rsplit('; ', 2)[2]
                        self.sentences.append(TaggedDocument(utils.to_unicode(sentence).split(),
                                                             [prefix + '_%s' % (item_no - 1)]))
            if prefix == "TRAIN_POS":
                numbers[0] = item_no - 1
            if prefix == "TRAIN_NEG":
                numbers[1] = item_no - 1
            if prefix == "CORPUS_2016":
                numbers[2] = item_no - 1
        return [self.sentences, numbers]


# better results if training phase is randomize (cf article)
    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

# oriane_pos + oriane_neg = 2015 and 2016.txt

def DataLoad(model, tConfig):
    sources = {tConfig['fOrianePos']:'TRAIN_POS', tConfig['fOrianeNeg']:'TRAIN_NEG', tConfig['fCorpus2016']:'CORPUS_2016'}

    print('TaggedTweet')
    xSentences = TaggedTweet(sources)

    # STEP 9
    print('D2V')
    results = xSentences.to_array()
    model.build_vocab(results[0])

    tConfig['number_oriane_pos'] = results[1][0]
    tConfig['number_oriane_neg'] = results[1][1]
    tConfig['number_tweet_2016'] = results[1][2]
    tConfig['train_number'] = results[1][0] + results[1][1]
    return [model, xSentences, tConfig]

def PrintDataInfo(tConfig):
    print('oriane_pos -> ' + str(tConfig['number_oriane_pos']))
    print('oriane_neg -> ' + str(tConfig['number_oriane_neg']))
    print('tweet_2016 -> ' + str(tConfig['number_tweet_2016']))


def Epoch(xSentences, model):
    print('Epoch')
    for epoch in range(10):
        print('EPOCH: {}'.format(epoch))
        token_count = sum([len(sentence) for sentence in xSentences])
        model.train(xSentences, total_examples = token_count, epochs = model.iter)
    return model

def ModelTraining(model, tConfig):
    train_arrays = numpy.zeros((tConfig['train_number'], 100))
    train_labels = numpy.zeros(tConfig['train_number'])

    print('Training Pos')
    for i in range(tConfig['number_oriane_pos']):
        prefix_train_pos = 'TRAIN_POS_' + str(i)
        train_arrays[i] = model.docvecs[prefix_train_pos]
        train_labels[i] = 1

    print('Training Neg')
    for i in range(tConfig['number_oriane_pos'], tConfig['number_oriane_pos'] + tConfig['number_oriane_neg'] - 1):
        prefix_train_neg = 'TRAIN_NEG_' + str(i - tConfig['number_oriane_pos'])
        train_arrays[i] = model.docvecs[prefix_train_neg]
        train_labels[i] = 0

    return [model, train_arrays, train_labels]

def Classifier(model, train_arrays, train_labels, tConfig):

    test_arrays = numpy.zeros((tConfig['number_tweet_2016'], 100))
    test_labels = numpy.zeros(tConfig['number_tweet_2016'])

    for i in range(tConfig['number_tweet_2016']):
        prefix_test = 'CORPUS_2016_' + str(i)
        test_arrays[i] = model.docvecs[prefix_test]


    # STEP 15
    print('Fitting')
    classifier = LogisticRegression()
    classifier.fit(train_arrays, train_labels)

    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

    print(classifier.score(test_arrays, test_labels))

    result_labels = classifier.predict(test_arrays)

    fResult = open(tConfig['sResultFile'], 'w', encoding='utf16')

    test_num = 1

    with open(tConfig['fCorpus2016'], "r", encoding='utf16') as file:
        for item_no, row in enumerate(file):
            if item_no >= 1 and result_labels[item_no - 2] == 1.:
                fResult.write(row)


def square_rooted(x):
    return round(sqrt(sum([a * a for a in x])), 3)


def cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return round(numerator / float(denominator), 3)

def jaccard_similarity(x, y):
    intersection_cardinality = len(set(x).intersection(set(y)))
    union_cardinality = len(set(x).union(set(y)))
    return intersection_cardinality / float(union_cardinality)

def dice_coefficient(a, b):
    """dice coefficient 2nt/na + nb."""
    a_bigrams = set(a)
    b_bigrams = set(b)
    overlap = len(a_bigrams & b_bigrams)
    return overlap * 2.0/(len(a_bigrams) + len(b_bigrams))

def ensemble(a, b):
    intersection_cardinality = len(set(a).intersection(set(b)))
    return intersection_cardinality / float(len(a))

def similarity(tConfig):
    fResult = open(tConfig['sResultFile'], 'r', encoding='utf16')
    fMabed = open(tConfig['sMabedFile'], 'r', encoding='utf16')

    tResult_s = []
    tMabed_s = []
    tResult_n = []
    tMabed_n = []

    print("file -> " + tConfig['sResultFile'])

    count = 0
    for line in fResult:
        tResult_s.append(line.split("; ", maxsplit=3)[2][1:-2])
        tResult_n.append(int(line.split("; ", maxsplit=3)[1]))
        count += 1

    count = 0
    for line in fMabed:
        tMabed_s.append(line.split("; ", maxsplit=3)[2][:-1])
        tMabed_n.append(int(line.split("; ", maxsplit=3)[1]))
        count += 1


    a = 0
    b = 0
    for x in tResult_n:
        b = 0
        for y in tMabed_n:
            if x == y:
                print(tResult_s[a])
                print(tMabed_s[b])
                print()
            b += 1
        a += 1

    #print(cosine_similarity(tResult, tMabed))

    print('TAILLE A')
    print(len(tResult_s))
    print('TAILLE B')
    print(len(tMabed_s))

    print('')
    print('tweets')
    print('')

    print('DICE')
    print(dice_coefficient(tResult_s, tMabed_s))
    print('JACCARD')
    print(jaccard_similarity(tResult_s, tMabed_s))

    print('A in B')
    print(ensemble(tResult_s, tMabed_s))
    print('B in A')
    print(ensemble(tMabed_s, tResult_s))

    print('')
    print('ID')
    print('')

    print('DICE')
    print(dice_coefficient(tResult_n, tMabed_n))
    print('JACCARD')
    print(jaccard_similarity(tResult_n, tMabed_n))

    print('A in B')
    print(ensemble(tResult_n, tMabed_n))
    print('B in A')
    print(ensemble(tMabed_n, tResult_n))

def main():
    model = Doc2Vec(min_count=1, window=5, size=100, sample=1e-4, negative=5, workers=7)
    tConfig = ConfigReader()
    ConfigPrint(tConfig)
    xResults = DataLoad(model, tConfig)
    model = xResults[0]
    xSentences = xResults[1]
    tConfig = xResults[2]
    # tConfig['number_oriane_pos'] = 1040
    # tConfig['number_oriane_neg'] = 29872
    # tConfig['number_tweet_2016'] = 475253
    # tConfig['train_number'] = 1040 + 29872
    # xSentences = ''

    PrintDataInfo(tConfig)
    if tConfig['sOptionLoad'] == '0':
        model = Epoch(xSentences, model)
        if tConfig['sOptionSave'] == '1':
            print("Save")
            model.save(tConfig['sOptionModelName'])
    else:
        print("Load")
        #tmpModel = ModelTest.ModelLoad(tConfig)
        model = Doc2Vec.load(tConfig['sOptionModelName'])

    xResults = ModelTraining(model, tConfig)
    model = xResults[0]
    train_arrays = xResults[1]
    train_labels = xResults[2]
    Classifier(model, train_arrays, train_labels, tConfig)
    similarity(tConfig)


if __name__ == '__main__':
    main()