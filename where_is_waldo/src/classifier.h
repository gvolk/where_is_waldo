#ifndef LOGREGCLASSIFIER_H
#define LOGREGCLASSIFIER_H


#include <cmath>
#include "feature.h"
#include "../src/gpu_classifier.h"

#define EPOCHS 100
#define LEARN_CONST .00001

#define CPU_MODE true
#define GPU_MODE false


//#define EPOCHS 100
//#define LEARN_CONST .0001

class LogRegClassifier
{

private:
    bool mode_cpu;

    double* beta_cpu;
    double* beta_gpu;

    double sigmoid(double z);

    double find_z( feature_data* training_data, int idx);

    void update_betaj(double* gradient);

    void report_accuracy(int correct_zeros, int correct_ones, int total_zeros, int total_ones);


public:
    LogRegClassifier(bool cpu_mode);

    //returns integer array of labels
    int* predict( feature_data* test_data);
    int* predict_cpu( feature_data* test_data);

    void train(feature_data* train_data);
    void train_cpu(feature_data* train_data);

    void test_classification(feature_data* test_data, feature_data* train_data);

    void set_mode(bool cpu_mode);


};

#endif // CLASSIFIER_H
