# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

# random shuffle
from random import shuffle

# numpy
import numpy

# classifier
from sklearn.linear_model import LogisticRegression

TRAIN_NUMBER = 32000
NUMBER_TWEET_2016 = 476032
NUMBER_TWEET_2016_POS = 0000
NUMBER_ORIANE_POS = 1048
NUMBER_ORIANE_NEG = 29970

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

# oriane_pos + oriane_neg = 2015 and 2016.txt
sources = {'oriane_positif.txt':'TRAIN_POS', 'oriane_negatif.txt':'TRAIN_NEG', '2016.txt':'TEST'}

print('TaggedTweet')
sentences = TaggedTweet(sources)

print('D2V')
model = Doc2Vec(min_count=1, window=5, size=100, sample=1e-4, negative=5, workers=7)
model.build_vocab(sentences.to_array())

print('Epoch')
for epoch in range(10):
    print('EPOCH: {}'.format(epoch))
    model.train(sentences.sentences_perm())

model.save('./imdb.d2v')
model = Doc2Vec.load('./imdb.d2v')

train_arrays = numpy.zeros((TRAIN_NUMBER, 100))
train_labels = numpy.zeros(TRAIN_NUMBER)

for i in range(NUMBER_ORIANE_POS):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_labels[i] = 1

for i in range(NUMBER_ORIANE_POS, NUMBER_ORIANE_POS + NUMBER_ORIANE_NEG):
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_neg]
    train_labels[i] = 0\

print(train_labels)

test_arrays = numpy.zeros((25000, 100))
test_labels = numpy.zeros(25000)

for i in range(NUMBER_TWEET_2016):
    prefix_test = 'TEST_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test]

print('Fitting')
classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

print(classifier.score(test_arrays, test_labels))