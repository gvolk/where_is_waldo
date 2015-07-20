#include "classifier.h"

LogRegClassifier::LogRegClassifier(bool cpu_mode)
{
    mode_cpu = cpu_mode;
    beta_cpu = new double[FEAT_LEN];
    beta_gpu = new double[FEAT_LEN];

    for(int i = 0; i< FEAT_LEN; i++)
    {
        beta_cpu[i] = (double)0;
        beta_gpu[i] = (double)0;
    }
}

void LogRegClassifier::set_mode(bool mode)
{
    mode_cpu = mode;
}

double LogRegClassifier::find_z(feature_data* training_data, int idx) {
    double z = 0;
    for(int i = 0; i < FEAT_LEN; i++) {
        z += beta_cpu[i] * training_data->features[idx*FEAT_LEN+ i];
    }
    return z;
}

void LogRegClassifier::update_betaj(double* gradient) {
    for(int i = 0; i < FEAT_LEN; i++) {
        beta_cpu[i] += LEARN_CONST * gradient[i];
    }
}


double LogRegClassifier::sigmoid(double z)
{
    return (1 / (1 + exp(-z)));
}

void LogRegClassifier::train_cpu(feature_data* training_data)
{
    double* beta_array = (double*) malloc(EPOCHS * FEAT_LEN * sizeof(double));
    std::pair<float,float> * p_corr = new std::pair<float, float>[EPOCHS * FEAT_LEN];
    for(int i = 0; i < EPOCHS; i++) {
        double gradient[FEAT_LEN] = {};
        for(int k = 0; k < training_data->num_pix_features; k++) {
            int output = training_data->labels[k];
            float z = find_z(training_data, k);
            float prob_y = sigmoid(z);

            for(int j = 0; j < FEAT_LEN; j++) {
                gradient[j] += (float)training_data->features[k*FEAT_LEN+j]*(output - prob_y);
            }

        }

        //qDebug() << ":" << beta[0]<< ":" << beta[1]<< ":" << beta[2]<< ":" << beta[3]<< ":" << beta[4]<< ":" << beta[5]<< ":" << beta[6]<< ":" << beta[7]<< ":" << beta[8] << "---- lik:" << lik;
        //qDebug()<<i;

        update_betaj(gradient);

        //save all betas to later choose the best
        p_corr[i] = calcPCorrect(training_data->labels, predict(training_data), training_data->num_pix_features);
        for(int j = 0; j < FEAT_LEN; j++) {

            beta_array[i*FEAT_LEN +j] = beta_cpu[j];
        }
    }

    //choose best beta
    bool break_outer =false;
    for(float f = 0.99 ; f > 0.5 ; f -=0.02)
    {
        for(int i = 0; i< EPOCHS; i++)
        {
            if(p_corr[i].first >= f && p_corr[i].second >= f)
            {
                for(int k = 0; k < FEAT_LEN; k++)
                {
                    beta_cpu[k] = beta_array[i*FEAT_LEN +k];
                }
                break_outer = true;
            }
        }
        if(break_outer)
        {
            break;
        }
    }
    free(beta_array);
}

void LogRegClassifier::train(feature_data* training_data)
{
    if(mode_cpu)
        train_cpu(training_data);
    else
    {
        train_cpu(training_data);
        qDebug() << ":" << beta_cpu[0]<< ":" << beta_cpu[1]<< ":" << beta_cpu[2]<< ":" << beta_cpu[3]<< ":" << beta_cpu[4]<< ":" << beta_cpu[5]<< ":" << beta_cpu[6]<< ":" << beta_cpu[7]<< ":" << beta_cpu[8];
        train_gpu(training_data->labels, training_data->features, training_data->num_pix_features, beta_gpu);
        qDebug() << ":" << beta_gpu[0]<< ":" << beta_gpu[1]<< ":" << beta_gpu[2]<< ":" << beta_gpu[3]<< ":" << beta_gpu[4]<< ":" << beta_gpu[5]<< ":" << beta_gpu[6]<< ":" << beta_gpu[7]<< ":" << beta_gpu[8];
    }
}


int* LogRegClassifier::predict_cpu(feature_data* test_data)
{
    int* labels = (int*)malloc(test_data->num_pix_features*sizeof(int));

    for(int i = 0; i < test_data->num_pix_features; i++) {
        double z = find_z(test_data, i);
        double prob_y = sigmoid(z);
        int estimated_class;
        if(prob_y > 0.5) estimated_class = 1;
        else estimated_class = 0;

        labels[i] = estimated_class;
    }
    return labels;
}

int* LogRegClassifier::predict(feature_data* test_data)
{
    if(mode_cpu)
        return predict_cpu(test_data);
    else
    {
        int* labels = (int*)malloc(test_data->num_pix_features*sizeof(int));
        predict_gpu(test_data->features, beta_cpu, test_data->num_pix_features, labels);
        return labels;
    }
}

std::pair<float,float> LogRegClassifier::calcPCorrect(int* labels, int* predicted, int num_features)
{
    float total_zeros = 0;
    float total_ones = 0;
    float correct_zeros = 0;
    float correct_ones = 0;

    for(int i = 0; i < num_features; i++) {
        if(labels[i] == predicted[i]) {
            if(labels[i] == 1)
            {
                correct_ones++;
            }
            else
            {
                correct_zeros++;
            }
        }
        if(labels[i] == 1)
        {
            total_ones++;
        }
        else
        {
            total_zeros++;
        }
    }

    return (std::make_pair((correct_ones/total_ones),(correct_zeros/ total_zeros)));
}


void LogRegClassifier::test_classification(feature_data* test, feature_data* train_data)
{
    int total_zeros = 0;
    int total_ones = 0;
    int correct_zeros = 0;
    int correct_ones = 0;
    int *labels = new int[train_data->num_pix_features];
    labels = predict(test);

    for(int i = 0; i < train_data->num_pix_features; i++) {
        if(train_data->labels[i] == labels[i]) {
            if(labels[i] == 1)
            {
                correct_ones++;
            }
            else
            {
                correct_zeros++;
            }
        }
        if(train_data->labels[i] == 1)
        {
            total_ones++;
        }
        else
        {
            total_zeros++;
        }
    }


    report_accuracy(correct_zeros, correct_ones, total_zeros, total_ones);
}


void LogRegClassifier::report_accuracy(int correct_zeros, int correct_ones, int total_zeros, int total_ones) {
    qDebug() << "Class 0: tested " << total_zeros << ", correctly classified " << correct_zeros;
    qDebug() << "Class 1: tested " << total_ones << ", correctly classified " << correct_ones;
    qDebug() << "Overall: tested " << total_zeros + total_ones << ", correctly classified " << correct_zeros + correct_ones;
    qDebug() << "Accuracy: = " << (float)(correct_zeros + correct_ones) / (total_zeros + total_ones);
}
