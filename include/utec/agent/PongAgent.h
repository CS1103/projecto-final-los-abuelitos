#ifndef PONGAGENT_H
#define PONGAGENT_H

#include <memory>
#include <array>

#include "utec/algebra/tensor.h"
#include "utec/nn/interfaces.h"
#include "utec/agent/EnvGym.h"

namespace utec::nn {

    using utec::algebra::Tensor;
    using utec::neural_network::ILayer;

    template <typename T>
    class PongAgent {
        std::unique_ptr<ILayer<T>> model;

    public:
        explicit PongAgent(std::unique_ptr<ILayer<T>> m) : model(std::move(m)) {}

        // Convierte un estado a un tensor y predice una acción
        int act(const State& s) {
            Tensor<T, 2> input(std::array<size_t, 2>{1, 3});
            input(0, 0) = static_cast<T>(s.ball_y);
            input(0, 1) = static_cast<T>(s.ball_x);
            input(0, 2) = static_cast<T>(s.paddle_y);

            Tensor<T, 2> output = model->forward(input);

            T up = output(0, 0);
            T stay = output(0, 1);
            T down = output(0, 2);

            if (up > stay && up > down) return -1;
            else if (down > up && down > stay) return +1;
            else return 0;
        }
    };

} // namespace utec::nn

#endif // PONGAGENT_H
