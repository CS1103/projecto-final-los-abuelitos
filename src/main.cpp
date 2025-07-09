#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <array>
#include <vector>

#include "utec/algebra/tensor.h"
#include "utec/nn/dense.h"
#include "utec/nn/activation.h"
#include "utec/nn/loss.h"
#include "utec/nn/neural_network.h"
#include "utec/agent/EnvGym.h"
#include "utec/agent/PongAgent.h"

using T = float;
using utec::algebra::Tensor;
using namespace utec::neural_network;
using namespace utec::nn;

// Carga entradas (x1, x2)
Tensor<T, 2> load_csv_X(const std::string& filename) {
    Tensor<T, 2> tensor(std::array<size_t, 2>{4, 2});
    std::ifstream file(filename);
    std::string line;
    size_t row = 0;
    while (std::getline(file, line) && row < 4) {
        std::stringstream ss(line);
        std::string x1, x2, y;
        std::getline(ss, x1, ',');
        std::getline(ss, x2, ',');
        std::getline(ss, y, ',');  // ignorar y
        tensor(row, 0) = std::stof(x1);
        tensor(row, 1) = std::stof(x2);
        row++;
    }
    return tensor;
}

// Carga salidas esperadas (y)
Tensor<T, 2> load_csv_Y(const std::string& filename) {
    Tensor<T, 2> tensor(std::array<size_t, 2>{4, 1});
    std::ifstream file(filename);
    std::string line;
    size_t row = 0;
    while (std::getline(file, line) && row < 4) {
        std::stringstream ss(line);
        std::string x1, x2, y;
        std::getline(ss, x1, ',');
        std::getline(ss, x2, ',');
        std::getline(ss, y, ',');
        tensor(row, 0) = std::stof(y);
        row++;
    }
    return tensor;
}

int main() {
    try {
        // Inicialización de pesos y bias
        auto w_init = [](Tensor<T, 2>& W) {
            for (auto& v : W.data()) v = static_cast<T>(0.1f);
        };
        auto b_init = [](Tensor<T, 2>& B) {
            for (auto& v : B.data()) v = static_cast<T>(0.0f);
        };

        // === Cargar datos desde xor_data.csv ===
        Tensor<T, 2> X = load_csv_X("xor_data.csv");
        Tensor<T, 2> Y = load_csv_Y("xor_data.csv");

        std::cout << "Shape de X: " << X.shape()[0] << " x " << X.shape()[1] << "\n";
        std::cout << "Shape de Y: " << Y.shape()[0] << " x " << Y.shape()[1] << "\n";

        // === Red Neuronal XOR ===
        NeuralNetwork<T> net;
        net.add_layer(std::make_unique<Dense<T>>(2, 4, w_init, b_init));
        net.add_layer(std::make_unique<ReLU<T>>());
        net.add_layer(std::make_unique<Dense<T>>(4, 1, w_init, b_init));

        net.train<MSELoss>(X, Y, 1000, 4, 0.1f);

        // === Mostrar predicciones ===
        std::cout << "\n=== Predicciones de la red XOR ===\n";
        Tensor<T, 2> predicciones = net.predict(X);
        for (size_t i = 0; i < predicciones.shape()[0]; ++i) {
            std::cout << "Input: (" << X(i, 0) << ", " << X(i, 1) << ") -> Predicho: "
                      << predicciones(i, 0) << " | Esperado: " << Y(i, 0) << "\n";
        }

        // === Simulación de entorno Pong (dummy) ===
        auto dummy_model = std::make_unique<Dense<T>>(3, 3, w_init, b_init);
        PongAgent<T> agent(std::move(dummy_model));

        EnvGym env;
        float reward;
        bool done = false;
        auto state = env.reset();

        for (int t = 0; t < 5; ++t) {
            int action = agent.act(state);
            state = env.step(action, reward, done);
            std::cout << "Step " << t << " -> reward: " << reward << ", done: " << done << "\n";
            if (done) break;
        }

    } catch (const std::out_of_range& e) {
        std::cerr << "Error: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Excepción inesperada: " << e.what() << "\n";
    }

    return 0;
}
