import numpy as np
from collections import defaultdict

'''
    The data is stored in a dict called dataSet, to use this class do the following:\
    d = DataReader('../data_set/letter.data')
    # dataset variable with following fields
    d.dataSet
    where filename = ../data_set/letter.data
    pass in the required filename
    Fields used:
    0.id: each letter is assigned a unique integer id
    1.letter: a-z
    2.next_id: id for next letter in the word, -1 if last letter
    3.fold: 0-9 -- cross-validation fold
    4.p_i_j: 0/1 -- value of pixel in row i, column j
'''

class DataReader():

    def __init__(self, filename):
        '''
        Sets up the required class variables from the parsed file.
        input : filename along with its path to be parsed
        Fields in actual .data file:
            1.id: each letter is assigned a unique integer id
            2.letter: a-z
            3.next_id: id for next letter in the word, -1 if last letter
            4.word_id: each word is assigned a unique integer id (not used)
            5.position: position of letter in the word (not used)
            6.fold: 0-9 -- cross-validation fold
            7.p_i_j: 0/1 -- value of pixel in row i, column j
        '''
        # filename to be parsed
        self.fileName = filename                       

    def parseFile(self):
        '''
        Parses the file name specifed in the constructor into a dictonary
        which has the feilds specified above
        '''
        data_set = {}        
        # opens the file to be parsed
        file_parse = open(self.fileName, 'r')
        
        # got through each line and add the lines to the dictionary
        for line_no, line in enumerate(file_parse):            
            # strip the line off tabs
            values = line.strip().split('\t')
            data = {}
            # index of the line
            data[0] = int(values[0])
            # lable of the line
            data[1] = values[1]
            # next word index
            data[2] = int(values[2])
            # cross validation fold used
            data[3] = int(values[5])
            # values of the pixels            
            pixel = np.array(list(map(int, values[6:])))
            # convert to an np array and reshape the pixel            
            pixel_np = pixel.reshape((16,8))
            data[4] = np.copy(pixel_np)
            
            # create a dictionary for each line                                
            data_set[line_no] = data                                                                        
                
        return data_set
    
    def build_test_words(self, data):
        test_words = []
        word = []
        for val in data:
            # val[2] is the id and val[3] is the next id 
            if (val[3] == val[2]+1):
                word.append(val[0])
            elif (val[3] == -1):
                word.append(val[0])                
                # val[3] is the cross validation fold
                test_words.append(word)
                word = []
        return test_words        
        
    def build_all_words(self, data):
        '''
        Groups the letters into words based on the letter id field 0 and 2        
        words = {}, Dictionary of words
        '''
        words = defaultdict(lambda : [])
        itr = 0
        for key, val in data.items():
            if (val[0] == key+1) and (val[2] == val[0]+1):
                words[itr].append(val[1])
            elif (val[0] == key+1) and (val[2] == -1):
                words[itr].append(val[1])
                itr += 1                
        return words        

if __name__ == "__main__":
    # read the given file
    d = DataReader('./letter.data')