import numpy as np
import data_reader
import random
from collections import defaultdict, Counter
from viterbi import Viterbi

class Naive_Bayes():
    '''
    Implements the Naive Bayes Classifier for character recognition.
    Implements HMM emission and transition probabilities
    '''
    def __init__(self,filename):
        '''
        self.data : Contains the parsed data as a dictionary
        self.
        self.theta_nb : Contains the weights for the naive bayes model which is nothing but log_pxy and log_py concatenated
                      for each label. Dimension of 1x129 - 1x128 for log_pxy ans 1x1 for log_py.
        '''
        self.dt = data_reader.DataReader(filename)
        self.data = self.dt.parseFile()
        self.theta_nb = defaultdict(float)
        self.labels = []
        self.START_LETTER = '**START**'
        self.END_LETTER = '**END**'
        self.smoothing = 0.1
        self.smooth_hmm = 0.1

    def group_data(self,data):
        cv_data = defaultdict(lambda : [])
        for key, val in data.items():
            # val[1] is the label
            label = val[1]
            self.labels.append(label)
            # val[4] are the pixel values
            pixels = val[4]
            # Flatten the 2d pixels into a single dimension vector of 128-dimenstions
            flatten_pixels = np.ndarray.flatten(pixels).reshape(1,-1)
            # val[0] is the id and val[2] is the next_id            
            # val[3] is the cross validation fold
            cv_data[val[3]].append((label,flatten_pixels,val[0],val[2]))
        return cv_data
    
    def cross_validation(self):                
        cv_data = self.group_data(self.data)        
        # The transition probabilities are done on the entire train and not on each fold.
        trans_probs = self.comp_transition_prob(self.data)        
        # Do 10-fold cross validation below.
        k = 10        
        for i in range(0,k):
            train_set = []
            valid_set = cv_data[i]            
            print("Validation Fold ",i+1)
            for j in range(0,k):
                if j != i:
                    train_set += cv_data[j]
            # Do the Naive Bayes Classification here
            self.estimate_nb(train_set)
            nb_pred_labels = self.predict(valid_set)                        
            nb_act_labels = [item[0] for item in valid_set]                        
            nb_acc = len(np.where(np.array(nb_pred_labels) == np.array(nb_act_labels))[0])
            print("Validation Accuracy of Naive Bayes ",nb_acc/len(nb_act_labels))
                        
            # The emission probabilities are done for each cv dataset.
            emission_probs = self.comp_emission_prob(nb_pred_labels,nb_act_labels)
            valid_words = self.dt.build_test_words(valid_set)            
            # Do the Viterbi step here            
            vt_pred_labels = []
            vt_act_labels = []
            nb_pred_labels = []
            itr = 0
            for w in valid_words:                
                nb_pred_word = self.predict(valid_set[itr:(itr+len(w))])
                nb_pred_labels += nb_pred_word
                vit = Viterbi(emission_probs,trans_probs,nb_pred_word)                
                vt_pred_labels += vit.hmmWord()                
                itr += len(w)
            nb_acc = len(np.where(np.array(nb_pred_labels) == np.array(nb_act_labels))[0])
            vt_acc = len(np.where(np.array(vt_pred_labels) == np.array(nb_act_labels))[0])                        
            print("Validation Accuracy of Viterbi ",vt_acc/len(nb_act_labels))                    

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
        for item in test_set:
            act_label = item[0]
            pixels = item[1]
            scores = dict.fromkeys(list(labels),0)
            for (key,val) in self.theta_nb.items():
                # Multiply the weights for the 128 pixels for each label with the features which is 
                # equivalent to adding the log weights
                scores[key] += np.sum(pixels * val[:-1])
                # Add the log prior probability
                scores[key] += val[-1]
            pred_label = self.argmax(scores)
            pred_labels.append(pred_label)                    
        return pred_labels

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
        s = self.smooth_hmm
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

    def comp_transition_prob(self,data):
        # First compute the transition counts from one letter to another
        # We will maintain a START LETTER constant for transitions into the first letter
        # We will maintain a END LETTER constant for transitions from the last letter
        trans_counts = defaultdict(lambda : Counter())
        words = self.dt.build_all_words(data)
        for key,val in words.items():
            trans_counts[self.START_LETTER].update([val[0]])
            for i in range(len(val)-1):
                trans_counts[val[i]].update([val[i+1]])
            trans_counts[val[-1]].update([self.END_LETTER])

        # Use the transition counts to compute the transition probabilities
        s = self.smooth_hmm
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

    