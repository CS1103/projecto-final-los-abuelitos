#ifndef ENVGYM_H
#define ENVGYM_H

namespace utec::nn {

    struct State {
        float ball_x, ball_y;
        float paddle_y;
    };

    class EnvGym {
    private:
        State state;
        float ball_dx;
        float ball_dy;
        float paddle_speed = 0.04f;
        float ball_speed = 0.02f;

    public:
        EnvGym();

        // Reinicia el entorno y devuelve el estado inicial
        State reset();

        // Aplica una acción, actualiza el estado y devuelve reward y done
        State step(int action, float& reward, bool& done);
    };

}

#endif // ENVGYM_H
