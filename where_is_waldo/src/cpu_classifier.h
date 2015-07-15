#ifndef CPU_CLASSIFIER_H
#define CPU_CLASSIFIER_H


#include <cmath>
#include "feature.h"

#define EPOCHS 100
#define LEARN_CONST .00001


//#define EPOCHS 100
//#define LEARN_CONST .0001

class CPU_Classifier
{

private:
    double* beta;

    double sigmoid(double z);

    double find_z( feature_data* training_data, int idx);

    void update_betaj(double* gradient);

    void report_accuracy(int correct_zeros, int correct_ones, int total_zeros, int total_ones);


public:
    CPU_Classifier();

    //returns integer array of labels
    int* predict( feature_data* test_data);

    void train(feature_data* train_data);

    void test_classification(feature_data* test_data, feature_data* train_data);


};

#endif // CLASSIFIER_H
