#include <iostream>
#include <vector>
#include <numeric>   
#include <fstream>   
#include <algorithm> 
#include <random>    
#include <tuple>    
#include <iomanip>   
#include "attention.h"
#include "feedforward.cpp"
#include "embedding.cpp" 
#include "functions.h"
#include "layernorm.h"
#include "multi_head_attention.cpp"
#include "linear_layer.h" 

using namespace std;

int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

// Carga las imágenes de MNIST
vector<vector<float>> load_mnist_images(string path) {
    ifstream file(path, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: No se pudo abrir el archivo de imágenes MNIST: " << path << endl;
        exit(1);
    }

    int magic_number = 0;
    int num_images = 0;
    int num_rows = 0;
    int num_cols = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    file.read((char*)&num_images, sizeof(num_images));
    num_images = reverseInt(num_images);
    file.read((char*)&num_rows, sizeof(num_rows));
    num_rows = reverseInt(num_rows);
    file.read((char*)&num_cols, sizeof(num_cols));
    num_cols = reverseInt(num_cols);

    int image_size = num_rows * num_cols;
    vector<vector<float>> dataset(num_images, vector<float>(image_size));

    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < image_size; ++j) {
            unsigned char pixel = 0;
            file.read((char*)&pixel, sizeof(pixel));
            // Normalizar a media 0 y desviación ~1
            dataset[i][j] = ((float)pixel - 127.5f) / 127.5f;
        }
    }
    file.close();
    return dataset;
}

// Carga las etiquetas de MNIST
vector<int> load_mnist_labels(string path) {
    ifstream file(path, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: No se pudo abrir el archivo de etiquetas MNIST: " << path << endl;
        exit(1);
    }

    int magic_number = 0;
    int num_labels = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    file.read((char*)&num_labels, sizeof(num_labels));
    num_labels = reverseInt(num_labels);

    vector<int> dataset(num_labels);
    for (int i = 0; i < num_labels; ++i) {
        unsigned char label = 0;
        file.read((char*)&label, sizeof(label));
        dataset[i] = (int)label;
    }
    file.close();
    return dataset;
}

// Convierte una etiqueta numérica a one-hot encoding
vector<float> to_one_hot(int label, int num_classes) {
    vector<float> one_hot_label(num_classes, 0.0f);
    if (label >= 0 && label < num_classes) {
        one_hot_label[label] = 1.0f;
    }
    return one_hot_label;
}

const int IMAGE_FLAT_SIZE = 28 * 28; 
const int NUM_CLASSES = 10;          

const int embedding_dim = 512;
const int num_heads = 8;
const int feedforward_dim = 2048;
const int num_layers = 6;
const float learning_rate = 0.001;
const int num_epochs = 5; 
const int batch_size = 32;

int main() {
    cout << fixed << setprecision(8);

    cout << "Cargando dataset MNIST..." << endl;
    vector<vector<float>> train_images = load_mnist_images("D:/UCSP/IA/Trabajo final/train-images.idx3-ubyte");
    vector<int> train_labels = load_mnist_labels("D:/UCSP/IA/Trabajo final/train-labels.idx1-ubyte");

    if (train_images.empty() || train_labels.empty()) {
        cerr << "No se cargaron imágenes o etiquetas de MNIST. Asegúrate de que las rutas de archivo son correctas y los archivos existen." << endl;
        return 1;
    }
    cout << "Dataset MNIST cargado. " << train_images.size() << " imágenes de entrenamiento." << endl;

    cout << "Inicializando componentes del Transformer..." << endl;
    // La capa de Embedding original ya no se usa.
    // Se reemplaza por una LinearLayer para proyectar la imagen aplanada al espacio de embedding.
    LinearLayer input_projection(IMAGE_FLAT_SIZE, embedding_dim, false); // No bias en esta proyección inicial

    vector<MultiHeadAttention> attention_layers;
    vector<Feedforward> feedforward_layers;

    for (int i = 0; i < num_layers; ++i) {
        attention_layers.emplace_back(num_heads, embedding_dim);
        feedforward_layers.emplace_back(embedding_dim, feedforward_dim);
    }

    // Capa de clasificación final (después de las capas Transformer)
    LinearLayer classification_head(embedding_dim, NUM_CLASSES, true);

    cout << "Componentes inicializados." << endl;

    // Generador de números aleatorios para barajar el dataset
    random_device rd;
    mt19937 g(rd());
    // Bucle de entrenamiento por épocas
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        cout << "\n--- Epoca " << epoch + 1 << " / " << num_epochs << " ---" << endl;

        // Barajar los índices del dataset para el entrenamiento por lotes
        vector<int> indices(train_images.size());
        iota(indices.begin(), indices.end(), 0); // Llenar con 0, 1, 2, ...
        shuffle(indices.begin(), indices.end(), g);

        float epoch_loss = 0.0f;
        int num_batches = (train_images.size() + batch_size - 1) / batch_size;
        int correct_predictions = 0;
        int total_samples = 0;

        for (int b_idx = 0; b_idx < num_batches; ++b_idx) {
            // --- Preparación del Lote ---
            vector<vector<float>> batch_input_images;
            vector<vector<float>> batch_target_labels_one_hot;

            int start_idx = b_idx * batch_size;
            int end_idx = min(start_idx + batch_size, (int)train_images.size());
            int current_batch_size = end_idx - start_idx;

            for (int i = start_idx; i < end_idx; ++i) {
                batch_input_images.push_back(train_images[indices[i]]);
                batch_target_labels_one_hot.push_back(to_one_hot(train_labels[indices[i]], NUM_CLASSES));
            }

            if (batch_input_images.empty()) continue;

            // --- Forward Pass ---
            vector<vector<float>> input_to_transformer = input_projection.forward(batch_input_images);

            vector<vector<float>> x_transformed = input_to_transformer;
            for (int i = 0; i < num_layers; ++i) {
                x_transformed = attention_layers[i].forward(x_transformed, x_transformed, x_transformed);
                x_transformed = feedforward_layers[i].forward(x_transformed);
            }

            vector<vector<float>> model_output = classification_head.forward(x_transformed);

            // --- Cálculo de Pérdida y Precisión ---
            vector<vector<float>> d_loss_batch(current_batch_size, vector<float>(NUM_CLASSES, 0.0f));
            float current_batch_loss = 0.0f;
            int batch_correct = 0;

            for (int i = 0; i < current_batch_size; ++i) {
                // Cálculo de pérdida
                float sample_loss;
                vector<float> sample_d_loss;
                tie(sample_loss, sample_d_loss) = softmaxLoss(model_output[i], batch_target_labels_one_hot[i]);
                current_batch_loss += sample_loss;
                d_loss_batch[i] = sample_d_loss;

                // Cálculo de precisión
                int predicted_class = max_element(model_output[i].begin(), model_output[i].end()) - model_output[i].begin();
                int true_class = max_element(batch_target_labels_one_hot[i].begin(), batch_target_labels_one_hot[i].end()) - batch_target_labels_one_hot[i].begin();
                if (predicted_class == true_class) {
                    batch_correct++;
                }
            }

            // Estadísticas del lote
            float avg_batch_loss = current_batch_loss / current_batch_size;
            float batch_accuracy = static_cast<float>(batch_correct) / current_batch_size * 100.0f;

            epoch_loss += avg_batch_loss;
            correct_predictions += batch_correct;
            total_samples += current_batch_size;

            // Mostrar progreso del lote
            cout << "  Batch " << b_idx + 1 << "/" << num_batches
                << " - Loss: " << fixed << setprecision(6) << avg_batch_loss
                << " - Acc: " << fixed << setprecision(2) << batch_accuracy << "%"
                << " (" << batch_correct << "/" << current_batch_size << ")"
                << endl;

            // --- Backward Pass ---
            vector<vector<float>> d_outputs = d_loss_batch;
            d_outputs = classification_head.backward(d_outputs, learning_rate, b_idx + 1);

            for (int i = num_layers - 1; i >= 0; --i) {
                d_outputs = feedforward_layers[i].backward(d_outputs, learning_rate, b_idx + 1);
                d_outputs = attention_layers[i].backward(d_outputs, learning_rate, b_idx + 1);
            }

            input_projection.backward(d_outputs, learning_rate, b_idx + 1);

        } // Fin del bucle de lotes

        // Estadísticas de la época
        float epoch_avg_loss = epoch_loss / num_batches;
        float epoch_accuracy = static_cast<float>(correct_predictions) / total_samples * 100.0f;

        cout << "  Epoch " << epoch + 1 << " Summary:"
            << " Avg Loss: " << fixed << setprecision(6) << epoch_avg_loss
            << " - Avg Acc: " << fixed << setprecision(2) << epoch_accuracy << "%"
            << " (" << correct_predictions << "/" << total_samples << ")"
            << endl;

    } // Fin del bucle de épocas

    cout << "\n--- Entrenamiento completo. ---" << endl;

    return 0;
}