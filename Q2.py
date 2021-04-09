import string
from string import digits
import math
import time


# evaluate the accuracy
def accuracy_metric(true_y, pred_y):
    correct = 0
    if len(true_y) != len(pred_y):
        return "Prediction length is different from Truth"
    for i in range(len(true_y)):
        if true_y[i] == pred_y[i]:
            correct += 1
    return correct / float(len(true_y)) * 100.0


# removing low tf-idf word from each sentence according to its label class in tf-idf
def preprocess(tfidf, dataset):
    newDataset = dataset
    for idx, row in enumerate(newDataset):
        sentence = row[3]
        label = row[2]
        words = sentence.split(" ")
        newDataset[idx][3] = ' '.join(w for w in [word for word in words if word in tfidf[label].keys()])
    return newDataset


# reading tsv file and save it in list
# remove punctuation
# remove digits
# transfer to lowercase
def read_tsv(tsv_file, data_type):
    tsv_lst = []
    with open(tsv_file) as f:
        lines = f.read().split('\n')[:-1]
        if data_type == "test":
            for line in lines:
                tsv_lst.append([line.split('\t')[0],
                                line.split('\t')[1],
                                line.split('\t')[2].translate(str.maketrans('', '', string.punctuation)).translate(
                                    str.maketrans('', '', digits)).lower()])
        else:
            for line in lines:
                tsv_lst.append([line.split('\t')[0],
                                line.split('\t')[1],
                                line.split('\t')[2],
                                line.split('\t')[3].translate(str.maketrans('', '', string.punctuation)).translate(
                                    str.maketrans('', '', digits)).lower()])

    return tsv_lst


# Calculate each label class' TF-IDF of dataset
class FeatureSelection(object):
    def __init__(self):
        # categories of label
        self.allClass = ["positive", "negative", "neutral"]
        # bag of word stored in dictionary
        self.bowDict = {"positive": [], "negative": [], "neutral": []}
        # using set to store unique words of dataset
        self.uniqueWords = set()
        # counting amount of words
        self.numOfWordsDict = {}

        # storing each class tf, idf, tf-idf in dictionary
        self.tf = {"positive": {}, "negative": {}, "neutral": {}}
        self.idf = {"positive": {}, "negative": {}, "neutral": {}}
        self.tfidf = {"positive": {}, "negative": {}, "neutral": {}}

    #################################################################################################################
    # The TF-IDF value increases proportionally to the number of times a word appears in the document and is offset #
    # by the number of documents in the corpus that contain the word, which helps to adjust for the fact that some  #
    # words appear more frequently in general.                                                                      #
    #################################################################################################################

    # calculating TF (term frequency)
    def computeTF(self):
        tfDict = {"positive": {}, "negative": {}, "neutral": {}}
        for c in self.allClass:
            # count of the category of bag of word
            bow_count = len(self.bowDict[c])
            # calculate word TF of each class
            for word, count in self.numOfWordsDict[c].items():
                # the number of times a word appears in a document divided by the total number of words in the document
                tfDict[c][word] = count / float(bow_count)
        return tfDict

    # calculating IDF (inverse document frequency)
    def computeIDF(self, docs):
        # length of all sentences
        N = len(docs)
        # initialize the dictionary of IDF
        idfDict = dict.fromkeys(docs[0].keys(), 0)
        for doc in docs:
            for word, val in doc.items():
                if val > 0:
                    idfDict[word] += 1

        for word, val in idfDict.items():
            # the log of the number of documents divided by the number of documents that contains the "word"
            idfDict[word] = math.log(N / float(val))

        return idfDict

    # calculating TF-IDF
    def computeTFIDF(self, dataset, n):
        # storing sentences of each label class
        sentencesDict = {"positive": [row[3] for row in dataset if row[2] == "positive"],
                         "negative": [row[3] for row in dataset if row[2] == "negative"],
                         "neutral": [row[3] for row in dataset if row[2] == "neutral"]}
        # splitting words of sentences in bag of word
        for c in self.allClass:
            for sent in sentencesDict[c]:
                for word in sent.split(" "):
                    self.bowDict[c].append(word)

        # getting unique words of whole dataset
        self.uniqueWords = set().union(self.bowDict["positive"], self.bowDict["negative"], self.bowDict["neutral"])

        # initialize the number of words dictionary
        self.numOfWordsDict = {"positive": dict.fromkeys(self.uniqueWords, 0),
                               "negative": dict.fromkeys(self.uniqueWords, 0),
                               "neutral": dict.fromkeys(self.uniqueWords, 0)}
        # counting word frequency of each label
        for c in self.allClass:
            for word in self.bowDict[c]:
                if word in self.uniqueWords:
                    self.numOfWordsDict[c][word] += 1
        # get TF
        self.tf = self.computeTF()
        # get IDF
        self.idf = self.computeIDF(docs=[self.numOfWordsDict[c] for c in self.allClass])
        # get TF-IDF
        for c in self.allClass:
            for word, val in self.tf[c].items():
                self.tfidf[c][word] = val * self.idf[word]

        # return top N TF-IDF word of each label class
        filteredTFIDF = {}
        for cls in self.tfidf.keys():
            filteredTFIDF[cls] = dict(list(
                {k: v for k, v in sorted(self.tfidf[cls].items(), key=lambda item: item[1], reverse=True)
                 if v > 0}.items())[:n])

        return filteredTFIDF


##########################################################################################################
# Multinomial NB = Multivariate NB, Multinomial Naive Bayes consider a feature vector where a given term #
# represents the number of times it appears or very often i.e. frequency.                                #
##########################################################################################################
# Multinomial Naive Bayes Classifier
class NaiveBayesClassifiers(object):
    def __init__(self):
        self.logPrior = {}
        self.logLikelihoods = {"positive": {}, "negative": {}, "neutral": {}}

        self.docs = {"positive": [], "negative": [], "neutral": []}
        self.vocab = []
        self.wordCount = {}
        self.all_classes = set()
        self.tfidf = {}

    # splitting words of all sentences
    def compute_vocab(self, sentences):
        vocabs = set()
        for s in sentences:
            for w in s.split(" "):
                vocabs.add(w)
        self.vocab = vocabs

    # counting each word in class
    def count_word_in_class(self):
        self.wordCount = {"positive": dict.fromkeys(self.vocab, 0),
                          "negative": dict.fromkeys(self.vocab, 0),
                          "neutral": dict.fromkeys(self.vocab, 0)}
        for c in self.wordCount.keys():
            docs = self.docs[c]
            for doc in docs:
                for word in doc.split(" "):
                    if word in self.vocab:
                        self.wordCount[c][word] += 1

    # training task
    def train(self, dataset, alpha, tfidf):
        self.tfidf = tfidf
        # size of dataset
        num_doc = len(dataset)

        # get sentences and sentiments' labels from dataset
        sentences = [row[3] for row in dataset]
        sentiments = [row[2] for row in dataset]

        # get all categories from dataset's label
        self.all_classes = set(sentiments)

        # splitting word of all sentences
        self.compute_vocab(sentences)

        # according each class to append sentences
        for s, c in zip(sentences, sentiments):
            self.docs[c].append(s)

        # counting word in class
        self.count_word_in_class()

        for c in self.all_classes:
            num_class = float(sentiments.count(c))
            # calculating the log prior probability
            # used for the prediction
            # counts of that class/number of documents
            self.logPrior[c] = math.log(num_class / num_doc)

            # word amount in current category
            total_count = 0
            for word in self.vocab:
                if word in self.wordCount[c].keys():
                    total_count += self.wordCount[c][word]

            # calculate log likelihoods of the word in current class
            # used for the prediction
            for word in self.vocab:
                count = self.wordCount[c][word]
                # log ( (frequency of word in current class + alpha) /
                # (word amount in current category + (alpha * word amount in whole dataset)) )
                self.logLikelihoods[c][word] = math.log((count + alpha) / (total_count + (alpha * len(self.vocab))))

    # sentiments of sentence prediction
    def predict(self, dataset):
        predictions = []
        sentences = [row[3] for row in dataset]

        for sent in sentences:
            sums = {"positive": 0, "negative": 0, "neutral": 0}
            for c in sums.keys():
                sums[c] = self.logPrior[c]
                words = sent.split(" ")
                for word in words:
                    # if the word appear in the training vocab list
                    if word in self.vocab:
                        # sum the log likelihoods of words of given sentences
                        sums[c] += self.logLikelihoods[c][word]
            # get the maximum posterior probability class
            predictions.append(max(sums, key=sums.get))
        return predictions


def main(alpha, n):
    # reading dataset
    train_tsv = read_tsv("./Assignment Data/Q2/Train.tsv", "train")
    val_tsv = read_tsv("./Assignment Data/Q2/Valid.tsv", "val")
    test_tsv = read_tsv("./Assignment Data/Q2//Test.tsv", "test")

    featureSelect = FeatureSelection()
    tfidf = featureSelect.computeTFIDF(train_tsv, n)
    train_tsv = preprocess(tfidf=tfidf, dataset=train_tsv)
    val_tsv = preprocess(tfidf=tfidf, dataset=val_tsv)

    NBClassifier = NaiveBayesClassifiers()
    NBClassifier.train(train_tsv, alpha, tfidf)

    print("Alpha: {}".format(alpha))
    print("N word of TF-IDF: {}".format(n))

    predictionLabel = NBClassifier.predict(train_tsv)
    trueLabel = [i[2] for i in train_tsv]
    print("Training Data Accuracy: {}".format(accuracy_metric(trueLabel, predictionLabel)))

    predictionLabel = NBClassifier.predict(val_tsv)
    trueLabel = [i[2] for i in val_tsv]
    print("Validation Data Accuracy: {}".format(accuracy_metric(trueLabel, predictionLabel)))


if __name__ == '__main__':
    start = time.time()
    # alpha - prevent zero counting (smoothing)
    alpha = [0.001]
    # top N tf-idf words in that class will be used for training and prediction
    N = [9000]
    for a in alpha:
        for n in N:
            main(a, n)
            print("")
    end = time.time()
    print("Running Time: {}".format(end - start))
