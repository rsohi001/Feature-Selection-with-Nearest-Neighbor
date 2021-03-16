import random
import numpy as np
import copy as cp

def leave_one_out_cross_val_forwards(data, current_set, feature_to_be_added):
    temp_data = cp.deepcopy(data)
    for i in current_set:
        for k in range(len(temp_data)):
            temp_data[k][i] = 0
    for i in range(len(temp_data)):
        temp_data[i][feature_to_be_added] = 0
    correct_classifications = 0
    data_size = len(temp_data)
    data_entry_len = len(temp_data[0])
    inf = float('inf')
    for i in range(data_size):
        object_properties = temp_data[i][1:]
        object_class = temp_data[i][0]

        NN_dist = inf
        NN_loc = inf

        for k in range(data_size):

            if k != i:
                distance = np.sqrt(sum(pow(object_properties - temp_data[k][1:], 2)))

                if distance < NN_dist:
                    NN_dist = distance
                    NN_loc = k
                    NN_class = data[k][0]

        if object_class == NN_class:
            correct_classifications += 1

    accuracy = correct_classifications / data_size
    return accuracy

def feature_search_demo_forwards(data):
    current_set_of_features = []
    length = len(data[0])
    ideal_set = []
    print_set = []
    best_accuracy = 0
    for i in range(1,length):
        print("On the ", i, "th level of the search tree")
        feature_to_be_added = []
        max_accuracy = 0
        for k in range(1,length):
            if k not in current_set_of_features:
                accuracy = leave_one_out_cross_val_forwards(data, current_set_of_features, k)
                print_set = cp.deepcopy(current_set_of_features)
                print_set.append(k)
                print("     Using feature(s)", print_set, "accuracy is ", round(accuracy*100,1), "%")

                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    feature_to_be_added = k

                if accuracy > best_accuracy:
                    best_accuracy = max_accuracy
                    ideal_set = cp.deepcopy(current_set_of_features)
                    ideal_set.append(k)

        current_set_of_features.append(feature_to_be_added)
        if(max_accuracy < best_accuracy):
            print("(Warning, accuracy has decreased! Continuing search in case of local maxima)")
        print("On level ", i, " feature ", feature_to_be_added, " was best with a accuracy of", round(max_accuracy*100,1), "%")
    print("Best accuracy:", round(best_accuracy*100,1), "% with set ", ideal_set)

def feature_search_demo_backwards(data):
    length = len(data[0])
    ideal_set = []
    best_accuracy = 0
    current_set_of_features = []
    for i in range(1, length):
        current_set_of_features.append(i)

    for i in range(1, length):
        print("On the ", i, "th level of the search tree")
        max_accuracy = 0
        for k in range(1, length):
            accuracy = leave_one_out_cross_val_backwards(data, current_set_of_features, k)
            print_set = cp.deepcopy(current_set_of_features)
            print_set.remove(k)
            print("     Using feature(s)", print_set, "accuracy is ", round(accuracy*100,1), "%")

            if accuracy > max_accuracy:
                max_accuracy = accuracy
                feature_to_be_removed = k

            if accuracy > best_accuracy:
                best_accuracy = max_accuracy
                ideal_set = cp.deepcopy(current_set_of_features)
                ideal_set.remove(k)

        current_set_of_features.remove(feature_to_be_removed)
        if(max_accuracy < best_accuracy):
            print("(Warning, accuracy has decreased! Continuing search in case of local maxima)")
        print("On level ", i, " feature ", feature_to_be_added, " was best with a accuracy of", round(max_accuracy*100,1), "%")

data = np.genfromtxt("CS170_SMALLtestdata__43.txt")
feature_search_demo_forwards(data)
