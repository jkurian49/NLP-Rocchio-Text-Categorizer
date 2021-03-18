#!/bin/bash

printf "corpus1_train.labels\ncorpus1_test.list\ncorpus1_predictions.labels\n" | python3 text_categorizer.py
echo -e "CORPUS 1 RESULTS\n"
perl analyze.pl corpus1_predictions.labels corpus1_test.labels


printf "corpus2_train.labels\ncorpus2_test.list\ncorpus2_predictions.labels\n" | python3 text_categorizer.py
echo -e "CORPUS 2 RESULTS"
perl analyze.pl corpus2_predictions.labels corpus2_test.labels

printf "corpus3_train.labels\ncorpus3_test.list\ncorpus3_predictions.labels\n" | python3 text_categorizer.py
echo -e "CORPUS 3 RESULTS"
perl analyze.pl corpus3_predictions.labels corpus3_test.labels
