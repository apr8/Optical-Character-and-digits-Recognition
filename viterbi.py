import numpy as np

class Viterbi():

    def __init__(self, emission, trans, word):
        '''
        input: emission
               trans
               word

        emission is an 26X26 dict storing the log-probability of observing word n, given a label l

        P(n|l) = emission[(n,l)]

        trans is an LxL dict storing the transition log-probability from the previous label (Yp) to the current label (Yc)

        P(Yc|Yp) = trans[(Yp,Yc)]

        start is a part of trans dict storing the transition log-probability of the beginning of a sentence <s> to every label l

        P(l|<s>) = trans[(start, l)]

        end is a Lx1 vector storing the transition log-probability from the label l of the last word to the end of the sentence </s>

        P(</s>|l) = trans[(l, end)]

        word is the list of observed charecters
        '''

        # define these as class variables
        self.emission = emission
        self.trans = trans
        self.word = word
        self.backpointer = {}
        self.hmm_word = {}

        # create an empty trellis (initialization)
        self.trellis = np.ones((26, len(word))) * -np.inf
        for count, i in enumerate(list(map(chr, range(97, 123)))):
            # add the initial start log probabilities
            #print self.word
            #print emission[(i, self.word[0])], trans[('start', i)]
            self.trellis[count, 0] = trans[('**START**', i)] + emission[(self.word[0], i)]

        self.recursionStep()
        #print(self.hmm_word, word)

    def hmmWord(self):
        out_word = []

        # loop over all the word and send as list
        for i in range(len(self.word)):
            # return the output word
            out_word.append(self.hmm_word[i])

        return out_word

    def recursionStep(self):
        '''
        recurcively loops over the trellis to find the path with maximum probability
        '''
        # loop over n time steps
        for i in range(len(self.word) - 1):
            # initialize and empty back pointer array
            self.backpointer[i] = {}
            # loop over each label to find the maximun trellis
            for count, j in enumerate(list(map(chr, range(97, 123)))):

                # loop over all the lables to calculate the max of trellis
                for c, l in enumerate(list(map(chr, range(97, 123)))):
                    # introducing a temp variable for each combination of trellis
                    tmp = self.trellis[c, i] + self.trans[(l, j)]

                    #print tmp, l, j, i, np.exp(self.trellis[count, i + 1])

                    # check if tmp is greater than previously stored trellis, if so replace
                    if tmp > self.trellis[count, i + 1]:
                        #print 'if', tmp, l, j, i, np.exp(self.trellis[count, i + 1])
                        self.trellis[count, i + 1] = tmp
                        # update the back pointer
                        self.backpointer[i][count] = l
                        #print l, self.backpointer
                # also add the emission probability at the end
                self.trellis[count, i + 1] += self.emission[(self.word[i + 1], j)]

        # calculate the maximum values
        lable_max, vit_max, lable_index = self.calculateMaxValue()
        self.hmm_word[len(self.word) - 1] = lable_max

        # do backpropagation and calculte the remaining n-1 word
        self.backpropagate(lable_index)
        return

    def backpropagate(self, lable_index):
        # copy the value to a local variable
        index = lable_index
        # back propagates over the back pointer
        i = len(self.word) - 2
        #print self.backpointer, self.trellis
        #print np.shape(self.trellis)
        while i >= 0:
            # back propagates from the maximum value(letter)

            letter = self.backpointer[i][index]
            #print 'letter',letter, i
            self.hmm_word[i] = letter
            #print letter, index

            # calculate the corresponding index
            index = ord(letter) - 97

            # decrement i
            i = i - 1

        return

    def calculateMaxValue(self):
        '''
        Calculate the maximum value and arg for the final step and returns the arg
        return : arg_max of Nth time
        '''
        # loop over all the letter finally and calculate the maximum value and the corresponding arg
        lable_max = 0
        lable_index = 0
        vit_max = -np.inf
        for count, j in enumerate(list(map(chr, range(97, 123)))):
            #check if its greater
            if self.trellis[count, len(self.word) - 1] + self.trans[(j, '**END**')] > vit_max:
                # update the max value and arg
                vit_max = self.trellis[count, len(self.word) - 1] + self.trans[(j, '**END**')]
                lable_max = j
                lable_index = count

        return lable_max, vit_max, lable_index
