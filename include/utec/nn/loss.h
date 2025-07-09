#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NN_LOSS_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NN_LOSS_H

#include "utec/nn/interfaces.h"
#include <cmath>

namespace utec::neural_network {

    // ============================ MSE Loss ============================
    template<typename T>
    struct MSELoss final : public ILoss<T, 2> {
        Tensor<T, 2> y_pred_, y_true_;

        MSELoss(const Tensor<T, 2>& y_pred, const Tensor<T, 2>& y_true)
                : y_pred_(y_pred), y_true_(y_true) {}

        T loss() const override {
            T sum = static_cast<T>(0);
            size_t total = y_pred_.shape()[0] * y_pred_.shape()[1];
            for (size_t i = 0; i < y_pred_.shape()[0]; ++i)
                for (size_t j = 0; j < y_pred_.shape()[1]; ++j) {
                    T diff = y_pred_(i, j) - y_true_(i, j);
                    sum += diff * diff;
                }
            return sum / static_cast<T>(total);
        }

        Tensor<T, 2> loss_gradient() const override {
            Tensor<T, 2> grad(y_pred_.shape());
            size_t total = y_pred_.shape()[0] * y_pred_.shape()[1];
            for (size_t i = 0; i < y_pred_.shape()[0]; ++i)
                for (size_t j = 0; j < y_pred_.shape()[1]; ++j)
                    grad(i, j) = (static_cast<T>(2) / static_cast<T>(total)) * (y_pred_(i, j) - y_true_(i, j));
            return grad;
        }
    };

    // ============================ BCE Loss ============================
    template<typename T>
    struct BCELoss final : public ILoss<T, 2> {
        Tensor<T, 2> y_pred_, y_true_;

        BCELoss(const Tensor<T, 2>& y_pred, const Tensor<T, 2>& y_true)
                : y_pred_(y_pred), y_true_(y_true) {}

        T loss() const override {
            T sum = static_cast<T>(0);
            size_t total = y_pred_.shape()[0] * y_pred_.shape()[1];
            for (size_t i = 0; i < y_pred_.shape()[0]; ++i)
                for (size_t j = 0; j < y_pred_.shape()[1]; ++j) {
                    T p = std::clamp(y_pred_(i, j), static_cast<T>(1e-7), static_cast<T>(1) - static_cast<T>(1e-7));
                    sum += -(y_true_(i, j) * std::log(p) + (static_cast<T>(1) - y_true_(i, j)) * std::log(static_cast<T>(1) - p));
                }
            return sum / static_cast<T>(total);
        }

        Tensor<T, 2> loss_gradient() const override {
            Tensor<T, 2> grad(y_pred_.shape());
            size_t total = y_pred_.shape()[0] * y_pred_.shape()[1];
            for (size_t i = 0; i < y_pred_.shape()[0]; ++i)
                for (size_t j = 0; j < y_pred_.shape()[1]; ++j) {
                    T p = std::clamp(y_pred_(i, j), static_cast<T>(1e-7), static_cast<T>(1) - static_cast<T>(1e-7));
                    grad(i, j) = (p - y_true_(i, j)) / (p * (static_cast<T>(1) - p));
                    grad(i, j) /= static_cast<T>(total);
                }
            return grad;
        }
    };

}

#endif // PROG3_NN_FINAL_PROJECT_V2025_01_NN_LOSS_H