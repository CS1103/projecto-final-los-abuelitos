#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NN_DENSE_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NN_DENSE_H

#include "utec/nn/interfaces.h"
#include <functional>
#include <array>
using utec::algebra::Tensor;

namespace utec::neural_network {

    template<typename T>
    struct Dense final : public ILayer<T> {
        Tensor<T, 2> W_, dW_;
        Tensor<T, 2> b_, db_;
        Tensor<T, 2> last_input_;

        // Constructor con inicializadores genéricos para W y b
        template<typename InitW, typename InitB>
        Dense(size_t in_features, size_t out_features, InitW init_w, InitB init_b) {
            std::array<size_t, 2> dims_W = {in_features, out_features};
            std::array<size_t, 2> dims_b = {1, out_features};
            W_ = Tensor<T, 2>(dims_W);
            b_ = Tensor<T, 2>(dims_b);
            dW_ = Tensor<T, 2>(dims_W);
            db_ = Tensor<T, 2>(dims_b);

            init_w(W_);
            init_b(b_);
        }

        Tensor<T, 2> forward(const Tensor<T, 2>& x) override {
            last_input_ = x;
            std::array<size_t, 2> dims_out = {x.shape()[0], W_.shape()[1]};
            Tensor<T, 2> output(dims_out);
            for (size_t i = 0; i < x.shape()[0]; ++i) {
                for (size_t j = 0; j < W_.shape()[1]; ++j) {
                    output(i, j) = b_(0, j);
                    for (size_t k = 0; k < W_.shape()[0]; ++k) {
                        output(i, j) += x(i, k) * W_(k, j);
                    }
                }
            }
            return output;
        }

        Tensor<T, 2> backward(const Tensor<T, 2>& gradients) override {
            size_t batch_size = gradients.shape()[0];

            // Calculamos dW
            for (size_t i = 0; i < W_.shape()[0]; ++i) {
                for (size_t j = 0; j < W_.shape()[1]; ++j) {
                    T sum = static_cast<T>(0);
                    for (size_t b = 0; b < batch_size; ++b) {
                        sum += last_input_(b, i) * gradients(b, j);
                    }
                    dW_(i, j) = sum / static_cast<T>(batch_size);
                }
            }

            // Calculamos db
            for (size_t j = 0; j < b_.shape()[1]; ++j) {
                T sum = static_cast<T>(0);
                for (size_t b = 0; b < batch_size; ++b) {
                    sum += gradients(b, j);
                }
                db_(0, j) = sum / static_cast<T>(batch_size);
            }

            // Gradiente hacia la entrada
            std::array<size_t, 2> dims_grad_input = {batch_size, W_.shape()[0]};
            Tensor<T, 2> grad_input(dims_grad_input);
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t i = 0; i < W_.shape()[0]; ++i) {
                    T sum = static_cast<T>(0);
                    for (size_t j = 0; j < W_.shape()[1]; ++j) {
                        sum += gradients(b, j) * W_(i, j);
                    }
                    grad_input(b, i) = sum;
                }
            }

            return grad_input;
        }

        void update_params(IOptimizer<T>& optimizer) override {
            optimizer.update(W_, dW_);
            optimizer.update(b_, db_);
        }
    };

}

#endif // PROG3_NN_FINAL_PROJECT_V2025_01_NN_DENSE_H