# Optical-Character-and-digits-Recognition
This is the final project for the course Probabilistic graphical models.

# Character Recognition

#### Below are the files for character recognition - <br />
  letter.data  (contains the OCR data used for character recognition) <br />
  data_reader.py (contains code for parsing the data) <br />
  naive_bayes.py (contains the Naive Bayes implementation) <br />
  viterbi.py (contains the Viterbi algorithm implementation) <br />
  main.py (to run the Character Recognition algorithm and get results) <br />
<br />
#### To run the Character Recogntion algorithms and get results please do the below - <br />
  python main.py <br />


# Digit Recognition

#### Below are the files containing the baseline logistic regression -
##### Source Code files -
  cal_te_acc.m <br />
  log_grad.m <br />
  logistic_classify.m <br />
  log_obj.m <br />
  log_reg.m <br />
  
##### Data files -
  usps_digital.mat <br />
  tr_X.txt <br />
  tr_y.txt <br />
  te_X.txt <br />
  te_y.txt <br />
  
#### To run the logistic regression algorithm for USPS digit data run the below file in matlab - <br />
  logistic_classify.m
  
#### Below are the files containing the Forward-Backward Greedy algorithm for leaning the Ising Model Structure -
##### Source Code files -
  digits_recognition_2.py  (contains the source code for learning structure of digit 2) <br />
  digits_recognition_3.py  (contains the source code for learning structure of digit 3) <br />
  digits_recognition.py (contains the general implementation of forward-backward algorithm) <br />
  digit_infer.py (contains source code for Gibbs Sampling which automatically runs the forward-backward algorithm)
  
#### To run the digit recognition algorithms and to get the learned structure 
##### 1. Run the forward-backward algorithm to learn the structure of the Ising Model
  python digits_recognition_3.py (for learning the structure of digit 3) <br />
  
  ###### This will create model_3_*.csv which contains the structure learned for digit 3
  ###### After the above file is created comment out all the lines from 452 to 475 in digits_recognition_3.py before doing the below 
  
##### 2. Run the Gibbs Sampling algorithm to do inference of the structure learned above
###### Modify line 6 of digit_infer.py shown below to include the model_3_*.csv file generated above 
###### weights_data = np.genfromtxt('model_3_*.csv', delimiter=',')
###### Now run the below -
  python digit_infer.py


  
