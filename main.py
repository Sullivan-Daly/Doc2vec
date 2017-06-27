# gensim modules

import warnings
import csv
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

# random shuffle
from random import shuffle

# numpy
import numpy

# classifier
from sklearn.linear_model import LogisticRegression

N_OPTION = 1
number_oriane_pos = 0
number_oriane_neg = 0
number_tweet_2016 = 0
train_number = 0

# FILE
F_ORIANE_POS = '../Data/test_csv.txt'
F_ORIANE_NEG = '../Data/oriane_neg_id-time-text.csv'
F_CORPUS_2016 = '../Data/corpus_2016_id-time-text.csv'

# LabeledSentence or TaggedDocument from gensim.models.doc2vec
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
sources = {F_ORIANE_POS:'TRAIN_POS', F_ORIANE_NEG:'TRAIN_NEG', F_CORPUS_2016:'CORPUS_2016'}

print('TaggedTweet')
sentences = TaggedTweet(sources)

# STEP 9
print('D2V')
model = Doc2Vec(min_count=1, window=5, size=100, sample=1e-4, negative=5, workers=7)
results = sentences.to_array()
model.build_vocab(results[0])

number_oriane_pos = results[1][0]
number_oriane_neg = results[1][1]
number_tweet_2016 = results[1][2]
train_number = number_oriane_pos + number_oriane_neg
print(number_oriane_pos)
print(number_oriane_neg)
print(number_tweet_2016)

if N_OPTION == 0:

    print('Epoch')
    for epoch in range(10):
        print('EPOCH: {}'.format(epoch))
        token_count = sum([len(sentence) for sentence in sentences])
        model.train(sentences, total_examples = token_count, epochs = model.iter)

    print("Save")
    model.save('tweets/test_csv.d2v')


print('Load')
model = Doc2Vec.load('tweets/test_csv.d2v')


# STEP 12
train_arrays = numpy.zeros((train_number, 100))
train_labels = numpy.zeros(train_number)

print('Training Pos')

print(len(model.docvecs))

for i in range(number_oriane_pos):
    prefix_train_pos = 'TRAIN_POS_' + str(i)



    test = model.docvecs['TRAIN_POS_0']


        #print()
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_labels[i] = 1

print('Training Neg')
for i in range(number_oriane_pos, number_oriane_pos + number_oriane_neg-1):
    prefix_train_neg = 'TRAIN_NEG_' + str(i-number_oriane_pos)
    train_arrays[i] = model.docvecs[prefix_train_neg]
    train_labels[i] = 0


test_arrays = numpy.zeros((number_tweet_2016, 100))
test_labels = numpy.zeros(number_tweet_2016)

for i in range(number_tweet_2016):
    prefix_test = 'CORPUS_2016_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test]

	
# STEP 15	
print('Fitting')
classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

		  
# STEP 17
		  
print(classifier.score(test_arrays, test_labels))
