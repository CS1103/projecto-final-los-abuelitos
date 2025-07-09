#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H

#include "utec/nn/interfaces.h"
#include "utec/nn/dense.h"
#include "utec/nn/loss.h"
#include "utec/nn/optimizer.h"
#include <vector>
#include <memory>
#include <stdexcept>  // <== Asegura manejo de errores

namespace utec::neural_network {

    template<typename T>
    struct NeuralNetwork {
        std::vector<std::unique_ptr<ILayer<T>>> layers_;

        void add_layer(std::unique_ptr<ILayer<T>> layer) {
            layers_.emplace_back(std::move(layer));
        }

        Tensor<T, 2> forward(const Tensor<T, 2>& input) {
            Tensor<T, 2> output = input;
            for (auto& layer : layers_)
                output = layer->forward(output);
            return output;
        }

        Tensor<T, 2> predict(const Tensor<T, 2>& input) {
            return forward(input);
        }

        void backward(const Tensor<T, 2>& gradients) {
            Tensor<T, 2> grad = gradients;
            for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i)
                grad = layers_[i]->backward(grad);
        }

        void update_all(IOptimizer<T>& optimizer) {
            for (auto& layer : layers_)
                layer->update_params(optimizer);
        }

        template<template<typename> class Loss>
        void train(const Tensor<T, 2>& X, const Tensor<T, 2>& Y,
                   size_t epochs, size_t batch_size, T learning_rate) {
            SGD<T> optimizer(learning_rate);
            for (size_t epoch = 0; epoch < epochs; ++epoch) {
                Tensor<T, 2> y_pred = forward(X);

                if (y_pred.shape() != Y.shape()) {
                    throw std::invalid_argument(
                        "Error: Las dimensiones de salida del modelo y las etiquetas no coinciden.\n" +
                        std::string("Prediccion: ") + std::to_string(y_pred.shape()[0]) + "x" + std::to_string(y_pred.shape()[1]) +
                        ", Etiquetas: " + std::to_string(Y.shape()[0]) + "x" + std::to_string(Y.shape()[1])
                    );
                }

                Loss<T> loss_function(y_pred, Y);
                T current_loss = loss_function.loss();
                Tensor<T, 2> gradients = loss_function.loss_gradient();
                backward(gradients);
                update_all(optimizer);
            }
        }
    };

}

#endif // PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
