#ifndef LAYERNORM_H
#define LAYERNORM_H

#include <vector>

class Layernorm {
private:
    float gamma;
    float beta;

    std::vector<float> mean;
    std::vector<float> variance;

    // Declaraciones añadidas para guardar las entradas y las entradas normalizadas
    // para el paso backward.
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> normalized_inputs;

public:
    Layernorm();

    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& inputs);

    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& d_outputs, float learning_rate, int t);
};

#endif // LAYERNORM_H