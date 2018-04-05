import numpy as np

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

        # Has all the seven feilds as specfied above
        self.dataSet = {}

        # parse the file
        self.parseFile()


    def parseFile(self):

        '''
        Parses the file name specifed in the constructor into a dictonary
        which has the feilds specified above
        '''

        # opens the file to be parsed
        file_parse = open(self.fileName, 'r')

        # got through each line and add the lines to the dictionary
        for line_no, line in enumerate(file_parse):


                # create a dictionary for each line
                self.dataSet[line_no] = {}

                # strip the line off tabs
                values = line.strip().split('\t')

                # test print
                #print type(line), line[2], type(values), int(values[0]), line_no

                # index of the line
                self.dataSet[line_no][0] = int(values[0])

                # lable of the line
                self.dataSet[line_no][1] = values[1]

                # next word index
                self.dataSet[line_no][2] = int(values[2])

                # cross validation fold used
                self.dataSet[line_no][3] = int(values[5])

                # values of the pixels
                pixel = np.array(map(int, values[6:]))

                # convert to an np array and reshape the pixel
                pixel_np = pixel.reshape((16,8))
                self.dataSet[line_no][4] = np.copy(pixel_np)

                print "line",line_no, self.dataSet[line_no]


        return

if __name__ == "__main__":

        # read the given file
        d = DataReader('../data_set/letter.data')
