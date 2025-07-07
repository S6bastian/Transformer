#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include "functions.h"
#include "matrix_operations.h"
#include "layernorm.h"

using namespace std;

// Constructor definition
Layernorm::Layernorm() {
    // Inicializa los miembros. No debes inicializarlos aquí con '='.
    // Se inicializan en el constructor o en la lista de inicialización.
    // Tu código original usa 'gamma = 1.0; beta = 1.0;', lo cual es incorrecto
    // dentro de la declaración de la clase en el .cpp.
    // Lo correcto es en la lista de inicialización o dentro del constructor.
    gamma = 1.0;
    beta = 1.0;
}

// Member function definitions
vector<vector<float>> Layernorm::forward(const vector<vector<float>>& inputs) {
    int batch_size = inputs.size();
    if (batch_size == 0) return {};
    int feature_size = inputs[0].size();

    // Save inputs for backward pass
    this->inputs = inputs;

    mean.assign(feature_size, 0.0f);
    variance.assign(feature_size, 0.0f);
    normalized_inputs.assign(batch_size, vector<float>(feature_size)); // Save normalized inputs

    for (int j = 0; j < feature_size; ++j) {
        for (int i = 0; i < batch_size; ++i) {
            mean[j] += inputs[i][j];
        }
        mean[j] /= batch_size;
    }

    for (int j = 0; j < feature_size; ++j) {
        for (int i = 0; i < batch_size; ++i) {
            variance[j] += pow(inputs[i][j] - mean[j], 2);
        }
        variance[j] /= batch_size;
    }

    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < feature_size; ++j) {
            normalized_inputs[i][j] = (inputs[i][j] - mean[j]) / sqrt(variance[j] + 1e-8);
            // Apply gamma and beta
            normalized_inputs[i][j] = gamma * normalized_inputs[i][j] + beta;
        }
    }

    // Return the output, which is the normalized and scaled input
    return normalized_inputs;
}

vector<vector<float>> Layernorm::backward(const vector<vector<float>>& d_outputs, float learning_rate, int t) {
    int batch_size = d_outputs.size();
    if (batch_size == 0) return {};
    int feature_size = d_outputs[0].size();

    vector<vector<float>> d_inputs(batch_size, vector<float>(feature_size));
    vector<float> d_gamma_sum(feature_size, 0.0f);
    vector<float> d_beta_sum(feature_size, 0.0f);

    // Step 1: Calculate gradients for gamma and beta
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < feature_size; ++j) {
            float x_hat = (inputs[i][j] - mean[j]) / sqrt(variance[j] + 1e-8);
            d_gamma_sum[j] += d_outputs[i][j] * x_hat;
            d_beta_sum[j] += d_outputs[i][j];
        }
    }

    // Step 2: Update gamma and beta
    for (int j = 0; j < feature_size; ++j) {
        gamma -= learning_rate * d_gamma_sum[j];
        beta -= learning_rate * d_beta_sum[j];
    }

    // Step 3: Calculate the gradient of the normalized input (x_hat)
    vector<vector<float>> d_x_hat(batch_size, vector<float>(feature_size));
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < feature_size; ++j) {
            d_x_hat[i][j] = d_outputs[i][j] * gamma;
        }
    }

    // Step 4: Calculate the gradient of the variance and mean
    vector<float> d_variance_sum(feature_size, 0.0f);
    vector<float> d_mean_sum(feature_size, 0.0f);

    for (int j = 0; j < feature_size; ++j) {
        float sqrt_variance_inv = 1.0f / sqrt(variance[j] + 1e-8);
        for (int i = 0; i < batch_size; ++i) {
            d_variance_sum[j] += d_x_hat[i][j] * (inputs[i][j] - mean[j]) * (-0.5f * pow(sqrt_variance_inv, 3));
        }
    }

    for (int j = 0; j < feature_size; ++j) {
        float sqrt_variance_inv = 1.0f / sqrt(variance[j] + 1e-8);
        for (int i = 0; i < batch_size; ++i) {
            d_mean_sum[j] += d_x_hat[i][j] * (-sqrt_variance_inv);
        }
    }

    // Step 5: Calculate the gradient of the input (x)
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < feature_size; ++j) {
            float sqrt_variance_inv = 1.0f / sqrt(variance[j] + 1e-8);
            d_inputs[i][j] = d_x_hat[i][j] * sqrt_variance_inv +
                d_variance_sum[j] * (2.0f * (inputs[i][j] - mean[j]) / batch_size) +
                d_mean_sum[j] / batch_size;
        }
    }

    return d_inputs;
}