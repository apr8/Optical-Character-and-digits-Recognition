#import digit_recognition_ups as dru
import digits_recognition_3 as dru
import numpy as np
import matplotlib.pyplot as plt

weights_data = np.genfromtxt('model_3_32.csv', delimiter=',')
# initialize all the pixel values
ising_model = dru.Graph()

def sample_pixels(p, v):
    # use the vertex as idx and calculate the probability of 1 or -1
    #prob = []
    #for i in [0,1]:
    #print v.vertex_id, weights_data[v.vertex_id, v.vertex_id], p, p[v.vertex_id, 1]
    prob = []
    pix = p
    prob.append(each_value(pix, v))

    pix[v.vertex_id - 1] = -1 * p[v.vertex_id - 1]
    prob.append(each_value(pix, v))

    return prob / np.sum(prob)

def each_value(p, v):
    #print v.vertex_id
    num = weights_data[v.vertex_id - 1, v.vertex_id - 1] * p[v.vertex_id - 1, 0]
    den = weights_data[v.vertex_id-1, v.vertex_id -1]
    for neigh in v.get_neighbors():
        #if neigh >= 1024:
        #    neigh = 1023
        #print neigh -1 , v.vertex_id, weights_data[v.vertex_id, neigh - 1]
        num += weights_data[v.vertex_id -1, neigh-1] * p[v.vertex_id -1, 0] * p[neigh-1, 0]
        den +=  weights_data[v.vertex_id - 1, neigh - 1] * p[v.vertex_id - 1, 0]

    prob = np.exp(num) / (1 + np.exp(den))

    return prob

def main():
    pixels = np.array(np.random.choice([-1, 1], size=1024))
    pixels = np.reshape(pixels, (1024,1))
    # initialize the list
    list_pixels = np.array(pixels)
    print weights_data.shape, pixels.shape

    # number of sweeps
    for i in range(0, 10):
        # loop over the number of pixels
        #rand_list = np.random.choice(range(1024),1024, replace=False)
        #for j in rand_list:
        print "eppoch", i
        for vert_id, vertex in enumerate(ising_model.vertices):

            # Draw the sample given all the disease assignments
            prob = sample_pixels(pixels, vertex)

            p = prob.flatten()
            #p = p / np.linalg.norm(p)
            value = [int(-1 * pixels[vert_id, 0]), int((pixels[vert_id, 0]))]
            #print prob, p,value
            # sample from this distribution
            rand = np.random.choice(value, p=p)
            #print rand, prob.tolist(), prob.shape
            # assign the sampled value
            pixels[vert_id, 0] = rand
            print 'data:', vertex.vertex_id - 1, rand, pixels[vert_id, 0]
            #if j == 49:
            #    print 'disease', np.transpose(disease), 'j',j

            list_pixels = np.append(np.copy(list_pixels), np.copy(pixels), axis = 1)

            #print 'ls:', list_disease, np.shape(list_disease)
            #print np.shape(list_disease)

    p = pixels.reshape((-1,32))
    plt.imshow(p)
    plt.show()
    # burn in the samples
    #print np.shape(list_disease)
    #list_disease = list_disease[:,1000:]
    #print np.shape(list_disease)
    ## select every 10th sample
    #sub_samp_list = list_disease[::,::10]

    ## count probabilities
    #sum_prob = np.sum(sub_samp_list, axis = 1)
    ##print 'numb',sum_prob
    ##print np.shape(sum_prob)
    ##print np.shape(sub_samp_list)

    #sum_prob = np.divide(sum_prob, float(np.shape(sub_samp_list)[1]))
    #print 'norm_prob_final = ', sum_prob

def predict(theta, train_data, test_data):
    test_results = []
    ising_model = dru.Graph(train_data)
    for test_id in range(len(test_data)):
        test_sample = test_data[test_id]
        theta_r = theta.diagonal()
        b_sum = np.sum(theta_r * test_sample)
        for vert_id, vertex in enumerate(ising_model.vertices):
            neigh = vertex.get_neighbors()
            theta_rt = theta[vert_id]
            theta_rt = theta_rt[neigh]
            a_sum = np.sum(theta_rt * test_sample[vert_id] * np.prod(test_sample[neigh]))

        p_x = np.exp(a_sum + b_sum)
        test_results.append(p_x)

    return test_results

def run():
    tr_data, tr_labels = dru.read_usps_data("tr_X.txt","tr_y.txt")
    te_data, te_labels = dru.read_usps_data("te_X.txt","te_y.txt")
    tr_data_len = int(len(tr_data)/10)
    te_data_len = int(len(te_data)/10)
    str1 = "model_ups_"
    str2 = ".csv"
    final_res = []
    for i in range(10):
        tr_start = i * tr_data_len
        te_start = i * te_data_len
        train_data = tr_data[tr_start:tr_start+tr_data_len]
        test_data = te_data[te_start:te_start+te_data_len]
        intm_res = []
        for j in range(10):
            file_name = str1 + str(i) + str2
            theta = np.genfromtxt(file_name,delimiter=',')
            results = predict(theta, train_data, test_data)
            intm_res.append(results)
        #res = np.array(intm_res)
        #final_res.append(np.argmax(res,axis=0))
        final_res.append(intm_res)
    return final_res

main()
