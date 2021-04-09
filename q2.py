import string
from string import digits
import math
import time


def accuracy_metric(true_y, pred_y):
    correct = 0
    if len(true_y) != len(pred_y):
        return "Prediction length is different from Truth"
    for i in range(len(true_y)):
        if true_y[i] == pred_y[i]:
            correct += 1
    return correct / float(len(true_y)) * 100.0


def preprocess(tfidf, dataset):
    newDataset = dataset
    for idx, row in enumerate(newDataset):
        sentence = row[3]
        label = row[2]
        words = sentence.split(" ")
        newDataset[idx][3] = ' '.join(w for w in [word for word in words if word in tfidf[label].keys()])
        # print("difference - origin: {}, new: {}".format(len(sentence), len(newDataset[idx][3])))
        # if len(sentence) == len(newDataset[idx][3]):
        #     print(sentence)
        #     print(newDataset[idx][3])

    return newDataset


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


class FeatureSelection(object):
    def __init__(self):
        # self.wordDict = {}
        self.allClass = ["positive", "negative", "neutral"]
        self.bowDict = {"positive": [], "negative": [], "neutral": []}
        self.uniqueWords = set()
        self.numOfWordsDict = {}

        self.tf = {"positive": {}, "negative": {}, "neutral": {}}
        self.idf = {"positive": {}, "negative": {}, "neutral": {}}
        self.tfidf = {"positive": {}, "negative": {}, "neutral": {}}

    def computeTF(self):
        tfDict = {"positive": {}, "negative": {}, "neutral": {}}
        for c in self.allClass:
            bow_count = len(self.bowDict[c])
            for word, count in self.numOfWordsDict[c].items():
                tfDict[c][word] = count / float(bow_count)
        return tfDict

    def computeIDF(self, docs):
        N = len(docs)
        idfDict = dict.fromkeys(docs[0].keys(), 0)
        for doc in docs:
            for word, val in doc.items():
                if val > 0:
                    idfDict[word] += 1
        for word, val in idfDict.items():
            idfDict[word] = math.log(N / float(val))

        return idfDict

    def computeTFIDF(self, dataset, n):
        sentencesDict = {"positive": [row[3] for row in dataset if row[2] == "positive"],
                         "negative": [row[3] for row in dataset if row[2] == "negative"],
                         "neutral": [row[3] for row in dataset if row[2] == "neutral"]}
        for c in self.allClass:
            for sent in sentencesDict[c]:
                for word in sent.split(" "):
                    self.bowDict[c].append(word)

        self.uniqueWords = set().union(self.bowDict["positive"], self.bowDict["negative"], self.bowDict["neutral"])

        self.numOfWordsDict = {"positive": dict.fromkeys(self.uniqueWords, 0),
                               "negative": dict.fromkeys(self.uniqueWords, 0),
                               "neutral": dict.fromkeys(self.uniqueWords, 0)}
        for c in self.allClass:
            for word in self.bowDict[c]:
                if word in self.uniqueWords:
                    self.numOfWordsDict[c][word] += 1

        self.tf = self.computeTF()
        self.idf = self.computeIDF(docs=[self.numOfWordsDict[c] for c in self.allClass])
        for c in self.allClass:
            for word, val in self.tf[c].items():
                self.tfidf[c][word] = val * self.idf[word]

        filteredTFIDF = {}
        for cls in self.tfidf.keys():
            filteredTFIDF[cls] = dict(list(
                {k: v for k, v in sorted(self.tfidf[cls].items(), key=lambda item: item[1], reverse=True)
                 if v > 0}.items())[:n])
        # for cls in self.tfidf.keys():
        #     print("Class: {} --- {}".format(cls, filteredTFIDF[cls]))

        # return self.tfidf
        return filteredTFIDF


class NaiveBayesClassifiers(object):
    def __init__(self):
        self.logPrior = {}
        self.logLikelihoods = {"positive": {}, "negative": {}, "neutral": {}}

        self.docs = {"positive": [], "negative": [], "neutral": []}
        self.vocab = []
        self.wordCount = {}
        self.all_classes = set()
        self.tfidf = {}

    def compute_vocab(self, sentences):
        vocabs = set()
        for s in sentences:
            for w in s.split(" "):
                vocabs.add(w)
        self.vocab = vocabs

    def count_word_in_class(self):
        self.wordCount = {"positive": dict.fromkeys(self.vocab, 0), "negative": dict.fromkeys(self.vocab, 0),
                          "neutral": dict.fromkeys(self.vocab, 0)}
        for c in self.wordCount.keys():
            docs = self.docs[c]
            for doc in docs:
                for word in doc.split(" "):
                    # if word in self.tfidf[c].keys():
                    if word in self.vocab:
                        self.wordCount[c][word] += 1

    # alpha ~ smoothing
    def train(self, dataset, alpha, tfidf):
        self.tfidf = tfidf
        num_doc = len(dataset)

        sentences = [row[3] for row in dataset]
        sentiments = [row[2] for row in dataset]
        self.all_classes = set(sentiments)

        self.compute_vocab(sentences)
        # newVocab = set()
        # for k, v in self.tfidf.items():
        #     for nested_key in self.tfidf[k].keys():
        #         newVocab.add(nested_key)
        # self.vocab = newVocab

        # according each class to append sentences
        for s, c in zip(sentences, sentiments):
            self.docs[c].append(s)

        self.count_word_in_class()

        for c in self.all_classes:
            num_class = float(sentiments.count(c))
            self.logPrior[c] = math.log(num_class / num_doc)
            total_count = 0
            for word in self.vocab:
                if word in self.wordCount[c].keys():
                    total_count += self.wordCount[c][word]

            for word in self.vocab:
                # if word in self.tfidf[c].keys():
                count = self.wordCount[c][word]
                self.logLikelihoods[c][word] = math.log((count + alpha) / (total_count + (alpha * len(self.vocab))))

    def predict(self, dataset):
        predictions = []
        sentences = [row[3] for row in dataset]

        for sent in sentences:
            sums = {"positive": 0, "negative": 0, "neutral": 0}
            for c in sums.keys():
                sums[c] = self.logPrior[c]
                words = sent.split(" ")
                for word in words:
                    if word in self.vocab:
                        sums[c] += self.logLikelihoods[c][word]
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
    alpha = [0.001]
    N = [9000]
    for a in alpha:
        for n in N:
            main(a, n)
            print("")
    end = time.time()
    print("Running Time: {}".format(end - start))
