#include <iostream>
#include <vector>
#include <cmath>
#include <tuple> // You need this for std::tuple
#include "functions.h"
#include "matrix_operations.h"
#include "attention.h" // <-- CRITICAL: Include your header file here

using namespace std;

// Constructor definition
ScaledDotProductAttention::ScaledDotProductAttention() {
    // Si no necesitas inicializar variables miembro de ScaledDotProductAttention
    // (ya que los std::vector se inicializan como vacíos por defecto),
    // el cuerpo puede estar vacío.
    // Ejemplo: queries.clear(); keys.clear(); // Opcional, ya se inicializan vacíos
}
// Member function definitions
void ScaledDotProductAttention::applyUpperTriangularMask(vector<vector<float>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (j > i) {
                matrix[i][j] = -INFINITY;
            }
        }
    }
}

vector<vector<float>> ScaledDotProductAttention::forward(const vector<vector<float>>& Q, const vector<vector<float>>& K, const vector<vector<float>>& V) {
    queries = Q;
    keys = K;
    values = V;
    vector<vector<float>> K_transpose;
    transpose(K, K_transpose);
    vector<vector<float>> QK;
    multiplyMatrices(Q, K_transpose, QK);

    float d_k = static_cast<float>(K[0].size());
    float scale_factor = sqrt(d_k);
    for (auto& row : QK) {
        for (auto& elem : row) {
            elem /= scale_factor;
        }
    }

    applyUpperTriangularMask(QK);
    softmax(QK);
    attention_weights = QK;
    multiplyMatrices(attention_weights, V, outputs);

    return outputs;
}

tuple<vector<vector<float>>, vector<vector<float>>, vector<vector<float>>> ScaledDotProductAttention::backward(const vector<vector<float>>& dL_dout) {
    vector<vector<float>> dL_dV, dL_dQ, dL_dK;

    vector<vector<float>> attention_weights_transpose;
    transpose(attention_weights, attention_weights_transpose);

    // Corrected calculation for dL_dV: (attention_weights)^T * dL_dout
    multiplyMatrices(attention_weights_transpose, dL_dout, dL_dV);

    vector<vector<float>> values_transpose;
    transpose(values, values_transpose);
    vector<vector<float>> dL_dattention_weights;
    // Corrected calculation for dL_dattention_weights: dL_dout * values^T
    multiplyMatrices(dL_dout, values_transpose, dL_dattention_weights);

    vector<vector<float>> dL_dQK = softmaxBackward(dL_dattention_weights, attention_weights);

    for (int i = 0; i < dL_dQK.size(); ++i) {
        for (int j = 0; j < dL_dQK[0].size(); ++j) {
            if (j > i) { // Mask the upper triangle for backward pass too
                dL_dQK[i][j] = 0;
            }
        }
    }

    float d_k = static_cast<float>(keys[0].size());
    float scale_factor = sqrt(d_k);
    for (auto& row : dL_dQK) {
        for (auto& elem : row) {
            elem /= scale_factor;
        }
    }

    // CORRECTION HERE for dL_dQ and dL_dK
    // dL_dQ = dL_dQK * K
    // Dimensions: (seq_len, seq_len) * (seq_len, head_dim) -> (seq_len, head_dim)
    multiplyMatrices(dL_dQK, keys, dL_dQ);

    // dL_dK = (dL_dQK)^T * Q
    // Dimensions: (seq_len, seq_len)^T * (seq_len, head_dim) -> (seq_len, seq_len) * (seq_len, head_dim) -> (seq_len, head_dim)
    vector<vector<float>> dL_dQK_transpose;
    transpose(dL_dQK, dL_dQK_transpose);
    multiplyMatrices(dL_dQK_transpose, queries, dL_dK);

    // dL_dV = (attention_weights)^T * dL_dout (already calculated above)

    return make_tuple(dL_dQ, dL_dK, dL_dV);
}