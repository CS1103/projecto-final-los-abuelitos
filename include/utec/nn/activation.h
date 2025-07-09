#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NN_ACTIVATION_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NN_ACTIVATION_H

#include "utec/nn/interfaces.h"
#include <cmath>

namespace utec::neural_network {

    // ============================ ReLU Activation ============================
    template<typename T>
    struct ReLU final : public ILayer<T> {
        Tensor<char, 2> mask_;

        Tensor<T, 2> forward(const Tensor<T, 2>& x) override {
            mask_ = Tensor<char, 2>(x.shape());
            Tensor<T, 2> output(x.shape());
            for (size_t i = 0; i < x.shape()[0]; ++i) {
                for (size_t j = 0; j < x.shape()[1]; ++j) {
                    output(i, j) = (x(i, j) > static_cast<T>(0)) ? x(i, j) : static_cast<T>(0);
                    mask_(i, j) = (x(i, j) > static_cast<T>(0)) ? 1 : 0;
                }
            }
            return output;
        }

        Tensor<T, 2> backward(const Tensor<T, 2>& gradients) override {
            Tensor<T, 2> grad_input(gradients.shape());
            for (size_t i = 0; i < gradients.shape()[0]; ++i) {
                for (size_t j = 0; j < gradients.shape()[1]; ++j) {
                    grad_input(i, j) = (mask_(i, j) ? gradients(i, j) : static_cast<T>(0));
                }
            }
            return grad_input;
        }
    };

    // ============================ Sigmoid Activation ============================
    template<typename T>
    struct Sigmoid final : public ILayer<T> {
        Tensor<T, 2> last_output_;

        Tensor<T, 2> forward(const Tensor<T, 2>& x) override {
            last_output_ = Tensor<T, 2>(x.shape());
            for (size_t i = 0; i < x.shape()[0]; ++i) {
                for (size_t j = 0; j < x.shape()[1]; ++j) {
                    last_output_(i, j) = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x(i, j)));
                }
            }
            return last_output_;
        }

        Tensor<T, 2> backward(const Tensor<T, 2>& gradients) override {
            Tensor<T, 2> grad_input(gradients.shape());
            for (size_t i = 0; i < gradients.shape()[0]; ++i) {
                for (size_t j = 0; j < gradients.shape()[1]; ++j) {
                    T sig = last_output_(i, j);
                    grad_input(i, j) = gradients(i, j) * sig * (static_cast<T>(1) - sig);
                }
            }
            return grad_input;
        }
    };

}

#endif // PROG3_NN_FINAL_PROJECT_V2025_01_NN_ACTIVATION_H
