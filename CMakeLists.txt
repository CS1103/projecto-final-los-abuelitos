cmake_minimum_required(VERSION 3.29)

project(projecto_final_los_abuelitos LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(projecto_final_los_abuelitos
        src/main.cpp
        src/main.cpp
        include/utec/algebra/Tensor.h
        include/utec/agent/PongAgent.h
        include/utec/nn/interfaces.h
        include/utec/nn/activation.h
        include/utec/nn/dense.h
        include/utec/nn/loss.h
        include/utec/nn/neural_network.h
        include/utec/nn/optimizer.h
        include/utec/agent/EnvGym.h
        src/utec/agent/EnvGym.cpp
)
