import numpy as np
import data_reader
from collections import defaultdict, Counter
from viterbi import Viterbi

class Naive_Bayes():
    '''
    Implements the Naive Bayes Classifier for character recognition.
    '''
    def __init__(self,filename):
        '''
        self.data : Contains the parsed data as a dictionary
        self.
        self.theta_nb : Contains the weights for the naive bayes model which is nothing but log_pxy and log_py concatenated
                      for each label. Dimension of 1x129 - 1x128 for log_pxy ans 1x1 for log_py.
        '''
        self.dt = data_reader.DataReader(filename)
        self.data = self.dt.dataSet
        self.theta_nb = defaultdict(float)
        self.labels = []
        self.START_LETTER = '**START**'
        self.END_LETTER = '**END**'
        self.smoothing = 0.01

    def cross_validation(self):
        data_set = defaultdict(lambda : [])
        # Group the dataset based on cross validation fold number
        for key, val in self.data.items():
            # val[1] is the label
            label = val[1]
            self.labels.append(label)
            # val[4] are the pixel values
            pixels = val[4]
            # Flatten the 2d pixels into a single dimension vector of 128-dimenstions
            flatten_pixels = np.ndarray.flatten(pixels).reshape(1,-1)
            # val[3] is the cross validation fold
            data_set[val[3]].append((label,flatten_pixels))

        # The transition probabilities are done on the entire dataset and not on each fold.
        trans_probs = self.comp_transition_prob()
        # Get the test words for viterbi
        test_words = self.dt.build_test_words()
        # Do 10-fold cross validation
        k = 10
        test_acc = []
        for i in range(0,k):
            train_set = []
            test_set = data_set[i]
            print("Fold ",i+1)
            for j in range(0,k):
                if j != i:
                    train_set += data_set[j]
            self.estimate_nb(train_set)
            acc, pred_labels = self.predict(test_set)
            act_labels = [item[0] for item in test_set]
            # The emission probabilities are done for each test dataset.
            emission_probs = self.comp_emission_prob(pred_labels,act_labels)            
            Viterbi(emission_probs,trans_probs,test_words[0][0])
            test_acc.append(acc)
        print("Final test accuracy ", np.sum(test_acc)/k)

    def proc_raw_feats(self,train_set):
        raw_feats = defaultdict(lambda : np.zeros([1,128],dtype=int))
        for item in train_set:
            # Get the label of this image
            label = item[0]
            # Get the pixels of this image
            pixels = item[1]
            val = raw_feats[label]
            # Concatenate the pixels of each image to form a nx128 dimension 2d array,
            #             where n = nos of images with this label
            # Remember to get rid of the 1st row which is initialized with 0s
            raw_feats[label] = np.concatenate((val,pixels),axis=0)

        return raw_feats

    def estimate_nb(self,train_set):
        '''
        raw_feats : Stores the features as a dictionary
                         Each 'key' is a label associated with the character
                         Each 'val' is a 2d-numpy array of dim(n,128 containing the pixels
                             for all the 'n' occurences of the label in the training data.
                         For efficiency we will create the dictionary with default np.array
                         filled with zeros and then concatenate the pixels.
                         Thus after processing the raw_feats omit the 1st row for all labels.
        feature_set : Stores the final counts of values for each 128 pixels of a label.
                           Its of dimension 1x128 for each label.
        '''
        raw_feats = self.proc_raw_feats(train_set)
        feature_set = {}
        for key, val in raw_feats.items():
            # Get rid of the 1st row
            val = val[1:,:]
            feature_set[key] = np.sum(val,axis=0)

        label_keys = np.array(self.labels)
        label_len = len(label_keys)
        '''
        log_pxy : Contains the log probability weights for each pixel in the image given the label.
        log_py : Contains the log probabilty for the prior for the label.
        '''
        log_pxy = defaultdict(float)
        s = self.smoothing
        for key, val in feature_set.items():
            counts_sum = np.sum(val)
            log_pxy[key] = [np.log((val[i] + s)/(counts_sum + (26 * s))) for i in range(0,128)]
            log_py = np.log(len(np.where(key == label_keys)[0])/label_len)
            self.theta_nb[key] = log_pxy[key] + [log_py]

    def argmax(self,scores):
        items = list(scores.items())
        items.sort()
        return items[np.argmax([i[1] for i in items])][0]

    def predict(self,test_set):
        pred_labels = []
        labels = set(self.labels)
        nos_incorrect = 0
        for item in test_set:
            act_label = item[0]
            pixels = item[1]
            scores = dict.fromkeys(list(labels),0)
            for (key,val) in self.theta_nb.items():
                # Multiply the weights for the 128 pixels for each label with the features
                scores[key] += np.sum(pixels * val[:-1])
                scores[key] += val[-1]
            pred_label = self.argmax(scores)
            pred_labels.append(pred_label)
            if act_label != pred_label:
                nos_incorrect += 1
        acc = nos_incorrect/len(test_set)
        print("Test Accuracy of Naive Bayes ",acc)
        return acc, pred_labels

    def comp_emission_prob(self,pred_labels,act_labels):
        # First compute the emission counts which is the nos of time character 'i'
        # is predicted as character 'j' by the Naive Bayes Classifier.
        act_labels = np.array(act_labels)
        pred_labels = np.array(pred_labels)
        emit_counts = defaultdict(lambda : Counter())
        for act_char in set(act_labels):
            indices = list(np.where(act_labels == act_char))[0]
            emit_counts[act_char].update(pred_labels[indices])

        # Use the emission counts to compute the emission probabilities
        s = self.smoothing
        emit_weights = defaultdict(float)
        all_letters = set(self.labels)
        for act_char in all_letters:
            # if the character is present in the emission counts
            if act_char in emit_counts:
                den = np.sum(list(emit_counts[act_char].values())) + (26 * s)
                for pred_char in all_letters:
                    num = emit_counts[act_char][pred_char] + s
                    emit_weights[(act_char,pred_char)] = np.log(num/den)
            # if the character is not present in the emission counts
            else:
                for pred_char in all_letters:
                    emit_weights[(act_char,pred_char)] = np.log((s)/(26 * s))

        return emit_weights

    def comp_transition_prob(self):
        # First compute the transition counts from one letter to another
        # We will maintain a START LETTER constant for transitions into the first letter
        # We will maintain a END LETTER constant for transitions from the last letter
        trans_counts = defaultdict(lambda : Counter())
        words = self.dt.build_all_words()
        for key,val in words.items():
            trans_counts[self.START_LETTER].update([val[0]])
            for i in range(len(val)-1):
                trans_counts[val[i]].update([val[i+1]])
            trans_counts[val[-1]].update([self.END_LETTER])

        # Use the transition counts to compute the transition probabilities
        s = self.smoothing
        trans_weights = defaultdict(float)
        all_letters = list(trans_counts.keys()) + [self.END_LETTER]
        V = len(list(trans_counts.keys()))
        for letter1 in all_letters:
            for letter2 in all_letters:
                if letter1 == self.END_LETTER or letter2 == self.START_LETTER:
                    trans_weights[(letter1,letter2)] = -np.inf
                else:
                    num = trans_counts[letter1][letter2] + s
                    den = sum(trans_counts[letter1].values()) + (V * s)
                    trans_weights[(letter1,letter2)] = np.log(num/den)

        return trans_weights

