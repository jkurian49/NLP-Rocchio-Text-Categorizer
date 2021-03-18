import sys
import os
from rocchio import Rocchio


def main():
    # prompt for input and output files
    train_file_set = input("Enter training file: ")
    test_file_set = input("Enter test file: ")
    output_file = input("Enter output file: ")
    
    # calculate centroids per class label
    train_data = Rocchio(documents=train_file_set,
                        train=True)
                        
    # use TFIDF statistics of training set to predict label of test files
    test_data = Rocchio(documents = test_file_set,
                        centroids = train_data.centroids,
                        base_TFIDF_vector = train_data.base_TFIDF_vector,
                        IDF = train_data.IDF,
                        avg_doc_length = train_data.avg_doc_length,
                        train=False)
                        
    # write predictions to file
    output_file = open(output_file,"w")
    for path,label in test_data.predictions.items():
        output_file.write(path + " " + label + "\n")
    

if __name__ == "__main__":
    main()
