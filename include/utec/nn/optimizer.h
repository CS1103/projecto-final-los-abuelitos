#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NN_OPTIMIZER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NN_OPTIMIZER_H

#include "utec/nn/interfaces.h"
#include <cmath>

namespace utec::neural_network {

    // ============================ SGD Optimizer ============================
    template<typename T>
    struct SGD final : public IOptimizer<T> {
        T learning_rate_;

        explicit SGD(T learning_rate) : learning_rate_(learning_rate) {}

        void update(Tensor<T, 2>& params, const Tensor<T, 2>& gradients) override {
            for (size_t i = 0; i < params.shape()[0]; ++i)
                for (size_t j = 0; j < params.shape()[1]; ++j)
                    params(i, j) -= learning_rate_ * gradients(i, j);
        }
    };

    // ============================ Adam Optimizer ============================
    template<typename T>
    struct Adam final : public IOptimizer<T> {
        T learning_rate_;
        T beta1_;
        T beta2_;
        T epsilon_;
        size_t t_ = 0;

        Tensor<T, 2> m_;
        Tensor<T, 2> v_;
        bool initialized_ = false;

        Adam(T learning_rate, T beta1 = static_cast<T>(0.9),
             T beta2 = static_cast<T>(0.999), T epsilon = static_cast<T>(1e-8))
                : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon) {}

        void update(Tensor<T, 2>& params, const Tensor<T, 2>& gradients) override {
            if (!initialized_) {
                m_ = Tensor<T, 2>(params.shape());
                v_ = Tensor<T, 2>(params.shape());
                m_.fill(static_cast<T>(0));
                v_.fill(static_cast<T>(0));
                initialized_ = true;
            }

            t_++;

            for (size_t i = 0; i < params.shape()[0]; ++i) {
                for (size_t j = 0; j < params.shape()[1]; ++j) {
                    m_(i, j) = beta1_ * m_(i, j) + (1 - beta1_) * gradients(i, j);
                    v_(i, j) = beta2_ * v_(i, j) + (1 - beta2_) * gradients(i, j) * gradients(i, j);

                    T m_hat = m_(i, j) / (1 - std::pow(beta1_, t_));
                    T v_hat = v_(i, j) / (1 - std::pow(beta2_, t_));

                    params(i, j) -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
                }
            }
        }
    };

}

#endif // PROG3_NN_FINAL_PROJECT_V2025_01_NN_OPTIMIZER_H
