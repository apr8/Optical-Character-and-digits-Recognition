import digit_recognition_ups as dru
import numpy as np

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
