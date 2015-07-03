#include "cpu_classifier.h"

CPU_Classifier::CPU_Classifier()
{
    beta = new double[FEAT_LEN];
}

double CPU_Classifier::find_z(feature_data* training_data, int idx) {
    double z = 0;
    for(int i = 0; i < FEAT_LEN; i++) {
        z += beta[i] * training_data->features[idx*FEAT_LEN+ i];
    }
    return z;
}

void CPU_Classifier::update_betaj(double* gradient) {
    for(int i = 0; i < FEAT_LEN; i++) {
        beta[i] += LEARN_CONST * gradient[i];
    }
}

void CPU_Classifier::train(feature_data* training_data) {
    for(int i = 0; i < EPOCHS; i++) {
        double gradient[FEAT_LEN] = {};

        for(int k = 0; k < training_data->num_pix_features; k++) {
            int output = training_data->labels[k];
            double z = find_z(training_data, k);
            for(int j = 0; j < FEAT_LEN; j++) {
                gradient[j] += (double)training_data->features[k*FEAT_LEN+j]*(output - 1 / (1 + exp(-z)));
            }
        }
        update_betaj(gradient);
    }
}

int* CPU_Classifier::predict(feature_data* test_data) {
    int labels[test_data->num_pix_features] = {};
    for(int i = 0; i < test_data->num_pix_features; i++) {
        double z = find_z(test_data, i);
        double prob_y = 1 / (1 + exp(-z));
        int estimated_class;
        if(prob_y > 0.5) estimated_class = 1;
        else estimated_class = 0;

        labels[i] = estimated_class;
    }
    return labels;
}

void CPU_Classifier::test_classification(feature_data* test, feature_data* train_data)
{
    int total_zeros = 0;
    int total_ones = 0;
    int correct_zeros = 0;
    int correct_ones = 0;
    int *labels = new int[train_data->num_pix_features];
    labels = predict(test);

    for(int i = 0; i < test->num_pix_features; i++) {
        if(test->labels[i] == labels[i]) {
            if(test->labels[i] == 1) {
                correct_ones++;
            }
            else {
                correct_zeros++;
            }
        }
        if(test->labels[i] == 1) total_ones++;
        else total_zeros++;
    }

    report_accuracy(correct_zeros, correct_ones, total_zeros, total_ones);
}

void CPU_Classifier::report_accuracy(int correct_zeros, int correct_ones, int total_zeros, int total_ones) {
    qDebug() << "Class -1: tested " << total_zeros << ", correctly classified " << correct_zeros;
    qDebug() << "Class 1: tested " << total_ones << ", correctly classified " << correct_ones;
    qDebug() << "Overall: tested " << total_zeros + total_ones << ", correctly classified " << correct_zeros + correct_ones;
    qDebug() << "Accuracy: = " << (double)(correct_zeros + correct_ones) / (total_zeros + total_ones);
}
