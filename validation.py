# gensim modules

import warnings
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


NUMBER_TEST = 300000
NUMBER_TEST_POS = 00000
NUMBER_POS = 100000
NUMBER_NEG = 200000
TRAIN_NUMBER = NUMBER_POS + NUMBER_NEG

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
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.dSources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

# better results if training phase is randomize (cf article)
    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

# validation
sources = {'validation/a_word.txt':'TRAIN_POS', 'validation/b_word.txt':'TRAIN_NEG', 'validation/test_word.txt':'TEST'}

print('TaggedTweet')
sentences = TaggedTweet(sources)

print('D2V')
model = Doc2Vec(min_count=1, window=5, size=100, sample=1e-4, negative=5, workers=7)
model.build_vocab(sentences.to_array())

print('Epoch')
for epoch in range(10):
    print('EPOCH: {}'.format(epoch))
    token_count = sum([len(sentence) for sentence in sentences])
    model.train(sentences, total_examples = token_count, epochs = model.iter)

model.save('validation/validation_test.d2v')
model = Doc2Vec.load('validation/validation_test.d2v')

train_arrays = numpy.zeros((TRAIN_NUMBER, 100))
train_labels = numpy.zeros(TRAIN_NUMBER)

for i in range(NUMBER_POS):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_labels[i] = 1

for i in range(NUMBER_POS, NUMBER_POS + NUMBER_NEG-1):
    prefix_train_neg = 'TRAIN_NEG_' + str(i-NUMBER_POS)
    train_arrays[i] = model.docvecs[prefix_train_neg]
    train_labels[i] = 0

print(train_arrays)
print(sentences)

test_arrays = numpy.zeros((NUMBER_TEST, 100))
test_labels = numpy.zeros(NUMBER_TEST)

for i in range(NUMBER_TEST):
    prefix_test = 'TEST_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test]

print('Fitting')
classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

print(classifier.score(test_arrays, test_labels))