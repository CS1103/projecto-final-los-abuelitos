#include "utec/agent/EnvGym.h"
#include <cstdlib> // para rand()
#include <ctime>   // para time()

namespace utec::nn {

    EnvGym::EnvGym() {
        std::srand(static_cast<unsigned>(std::time(nullptr)));
    }

    State EnvGym::reset() {
        state.ball_x = 0.5f;
        state.ball_y = 0.5f;
        state.paddle_y = 0.5f;
        ball_dx = ((std::rand() % 2) ? 1 : -1) * ball_speed;
        ball_dy = ((std::rand() % 2) ? 1 : -1) * ball_speed;
        return state;
    }

    State EnvGym::step(int action, float& reward, bool& done) {
        // Mueve la paleta
        if (action == -1) state.paddle_y -= paddle_speed;
        else if (action == 1) state.paddle_y += paddle_speed;

        // Clampa la paleta
        if (state.paddle_y < 0.0f) state.paddle_y = 0.0f;
        if (state.paddle_y > 1.0f) state.paddle_y = 1.0f;

        // Mueve la bola
        state.ball_x += ball_dx;
        state.ball_y += ball_dy;

        // Rebote en paredes superior e inferior
        if (state.ball_y <= 0.0f || state.ball_y >= 1.0f)
            ball_dy = -ball_dy;

        // Rebote en el lado izquierdo
        if (state.ball_x <= 0.0f)
            ball_dx = -ball_dx;

        // Colisión con la paleta (lado derecho)
        reward = 0.0f;
        done = false;

        if (state.ball_x >= 1.0f) {
            if (std::abs(state.ball_y - state.paddle_y) <= 0.1f) {
                // Éxito: devuelve la bola
                reward = +1.0f;
                ball_dx = -ball_dx;
            } else {
                // Falla: punto perdido
                reward = -1.0f;
                done = true;
            }
        }

        return state;
    }

}
