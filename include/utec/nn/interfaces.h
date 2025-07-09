#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NN_INTERFACES_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NN_INTERFACES_H

#include "utec/algebra/tensor.h"
using utec::algebra::Tensor;

namespace utec::neural_network {


    // ============================ Optimizer Interface ============================
    template<typename T>
    struct IOptimizer {
        virtual ~IOptimizer() = default;
        virtual void update(Tensor<T,2>& params, const Tensor<T,2>& gradients) = 0;
        virtual void step() {}
    };

    // ============================ Layer Interface ============================
    template<typename T>
    struct ILayer {
        virtual ~ILayer() = default;
        virtual Tensor<T,2> forward(const Tensor<T,2>& x) = 0;
        virtual Tensor<T,2> backward(const Tensor<T,2>& gradients) = 0;
        virtual void update_params(IOptimizer<T>& optimizer) {}
    };

    // ============================ Loss Interface ============================
    template<typename T, size_t DIMS>
    struct ILoss {
        virtual ~ILoss() = default;
        virtual T loss() const = 0;
        virtual Tensor<T,DIMS> loss_gradient() const = 0;
    };

}

#endif // PROG3_NN_FINAL_PROJECT_V2025_01_NN_INTERFACES_H
