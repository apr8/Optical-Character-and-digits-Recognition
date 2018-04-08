# Optical-Character-and-digits-Recognition
This is the final project for the course Probabilistic graphical models.

# Using Naive Bayes

import naive_bayes

nb = naive_bayes.Naive_Bayes("./letter.data")
nb.cross_validation()

emission_probs : dict{(Character of word, Any of the Character 1 to 26) : Emission Probability}
trans_probs : dict{(Character 1, Character 2) : Transition Probability}
test_words : List of words to be predicted
