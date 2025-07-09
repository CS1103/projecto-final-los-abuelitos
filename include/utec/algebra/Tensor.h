#ifndef PROG3_TENSOR_FINAL_PROJECT_V2025_01_TENSOR_H
#define PROG3_TENSOR_FINAL_PROJECT_V2025_01_TENSOR_H

#include <array>
#include <vector>
#include <cassert>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <string>
#include <functional>

namespace utec {
    namespace algebra {

        template <typename T, size_t N>
        class Tensor {
            static_assert(N >= 1);

        public:
            using shape_ = std::array<size_t, N>;
            using size_ = size_t;
            using value_ = T;

            // Constructor desde std::array (forma segura y directa)
            explicit Tensor(const shape_& dims) : dims_(dims) {
                _initialize();
            }

            template <typename... Dims>
                explicit Tensor(Dims... dims) {
                if (sizeof...(Dims) != N)
                    throw std::invalid_argument("Number of dimensions do not match with " + std::to_string(N));
                size_t vals[] = { static_cast<size_t>(dims)... };
                for (size_t i = 0; i < N; ++i) dims_[i] = vals[i];
                _initialize();
            }


            // Constructores por copia y movimiento
            Tensor() {
                dims_.fill(0);
                strides_.fill(0);
                total_size_ = 0;
            }

            Tensor(const Tensor&) = default;
            Tensor(Tensor&&) noexcept = default;
            Tensor& operator=(const Tensor&) = default;
            Tensor& operator=(Tensor&&) noexcept = default;
            ~Tensor() = default;


            //  Acceso
            shape_ shape() const noexcept { return dims_; }
            size_ size() const noexcept { return total_size_; }
            const std::vector<T>& data() const noexcept { return data_; }
            std::vector<T>& data() noexcept { return data_; }

            auto begin() noexcept { return data_.begin(); }
            auto end() noexcept { return data_.end(); }
            auto begin() const noexcept { return data_.cbegin(); }
            auto end() const noexcept { return data_.cend(); }
            auto cbegin() const noexcept { return data_.cbegin(); }
            auto cend() const noexcept { return data_.cend(); }

            void fill(const T& value) {
                std::fill(data_.begin(), data_.end(), value);
            }

            // Acceso por índices
            template <typename... Idx>
            T& operator()(Idx... idx) {
                if (sizeof...(Idx) != N)
                    throw std::invalid_argument("Number of dimensions do not match with 2");
                std::array<size_t, N> arr = { static_cast<size_t>(idx)... };
                return data_[_flat_index(arr)];
            }

            template <typename... Idx>
            const T& operator()(Idx... idx) const {
                if (sizeof...(Idx) != N)
                    throw std::invalid_argument("Number of dimensions do not match with 2");
                std::array<size_t, N> arr = { static_cast<size_t>(idx)... };
                return data_[_flat_index(arr)];
            }

            Tensor& operator=(std::initializer_list<T> ilist) {
                if (ilist.size() != total_size_)
                    throw std::invalid_argument("Data size does not match tensor size");
                std::copy(ilist.begin(), ilist.end(), data_.begin());
                return *this;
            }

            // Cambiar dimensiones
            template <typename... Dims>
                void reshape(Dims... new_dims) {
                constexpr size_t count = sizeof...(Dims);
                if (count != N)
                    throw std::invalid_argument("Number of dimensions do not match with " + std::to_string(N));

                size_t vals[] = { static_cast<size_t>(new_dims)... };
                shape_ nd;
                size_t new_total = 1;
                for (size_t i = 0; i < N; ++i) {
                    if ((nd[i] = vals[i]) == 0)
                        throw std::invalid_argument("ERROR: Zero dimension in reshape");
                    new_total *= nd[i];
                }

                dims_ = nd;
                total_size_ = new_total;
                _recompute_strides();
                data_.resize(new_total);
            }


            // Mostrar tensor (solo para fines visuales)
            friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    if (t.total_size_ == 0) return os << "(empty tensor)\n";

    if constexpr (N == 1) {
        os << "{ ";
        for (size_t i = 0; i < t.dims_[0]; ++i) {
            os << t.data_[i];
            if (i + 1 < t.dims_[0]) os << " ";
        }
        os << "\n";

    }
    else if constexpr (N == 2) {
        os << "{\n";
        for (size_t i = 0; i < t.dims_[0]; ++i) {
            for (size_t j = 0; j < t.dims_[1]; ++j) {
                os << t(i, j);
                if (j + 1 < t.dims_[1]) os << " ";
            }
            os << "\n";
        }
        os << "}";
    }


                else {
                    std::vector<size_t> indices(N, 0);

                    if constexpr (N == 3) {
                        os << "{\n";
                        for (size_t i = 0; i < t.dims_[0]; ++i) {
                            os << "{\n";
                            for (size_t j = 0; j < t.dims_[1]; ++j) {
                                os << " ";
                                for (size_t k = 0; k < t.dims_[2]; ++k) {
                                    size_t flat = i * t.strides_[0] + j * t.strides_[1] + k * t.strides_[2];
                                    os << t.data_[flat];
                                    if (k + 1 < t.dims_[2]) os << " ";
                                }
                                os << "\n";
                            }
                            os << "}";
                            if (i + 1 < t.dims_[0]) os << "\n";
                        }
                        os << "\n}";
                    } else {
                        std::function<void(size_t, std::vector<size_t>&)> print_recursive;
                        print_recursive = [&](size_t dim, std::vector<size_t>& indices) {
                            if (dim == N - 2) {
                                os << "{\n";
                                for (size_t i = 0; i < t.dims_[dim]; ++i) {
                                    indices[dim] = i;
                                    os << "{ ";
                                    for (size_t j = 0; j < t.dims_[dim + 1]; ++j) {
                                        indices[dim + 1] = j;
                                        size_t flat = 0;
                                        for (size_t k = 0; k < N; ++k)
                                            flat += indices[k] * t.strides_[k];
                                        os << t.data_[flat];
                                        if (j + 1 < t.dims_[dim + 1]) os << " ";
                                    }
                                    os << " }";
                                    if (i + 1 < t.dims_[dim]) os << "\n";
                                }
                                os << "\n}";
                            }
                            else {
                                os << "{\n";
                                for (size_t i = 0; i < t.dims_[dim]; ++i) {
                                    indices[dim] = i;
                                    print_recursive(dim + 1, indices);
                                    if (i + 1 < t.dims_[dim]) os << "\n";
                                }
                                os << "\n}";
                            }
                        };

                        os << "{\n";
                        print_recursive(0, indices);
                        os << "\n}";
                    }
                }

    return os;
}



            const shape_& dims() const noexcept { return dims_; }
            const shape_& strides() const noexcept { return strides_; }

        private:
            shape_ dims_{};
            shape_ strides_{};
            size_ total_size_{};
            std::vector<T> data_;

            void _initialize() {
                total_size_ = 1;
                for (const auto& d : dims_) {
                    if (d == 0) throw std::invalid_argument("Tensor dimensions must be > 0");
                    total_size_ *= d;
                }
                _recompute_strides();
                data_.resize(total_size_);
            }

            void _recompute_strides() {
                strides_[N - 1] = 1;
                for (int i = int(N) - 2; i >= 0; --i) {
                    strides_[i] = strides_[i + 1] * dims_[i + 1];
                }
            }

            size_t _flat_index(const std::array<size_t, N>& idx) const {
                size_t lin = 0;
                for (size_t i = 0; i < N; ++i) {
                    if (idx[i] >= dims_[i])
                        throw std::out_of_range("Index out of bounds");
                    lin += idx[i] * strides_[i];
                }
                return lin;
            }
        };

// FUNCIONES AUXILIARES Y OPERADORES

// Broadcasting entre dos tensores
        template <typename T, size_t N>
        std::array<size_t, N> _compute_broadcast_shape(const Tensor<T, N>& A, const Tensor<T, N>& B) {
            std::array<size_t, N> result;
            for (size_t i = 0; i < N; ++i) {
                if (A.dims()[i] == B.dims()[i]) result[i] = A.dims()[i];
                else if (A.dims()[i] == 1) result[i] = B.dims()[i];
                else if (B.dims()[i] == 1) result[i] = A.dims()[i];
                else throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");
            }
            return result;
        }

        // Función genérica para operaciones binarias con broadcasting
        template <typename T, size_t N, typename BinaryOp>
        Tensor<T, N> _binary_op(const Tensor<T, N>& A, const Tensor<T, N>& B, BinaryOp op) {
            auto C_dims = _compute_broadcast_shape(A, B);
            Tensor<T, N> C(C_dims);
            for (size_t lin = 0; lin < C.size(); ++lin) {
                std::array<size_t, N> idxC;
                size_t rem = lin;
                for (int i = N - 1; i >= 0; --i) {
                    idxC[i] = rem % C.dims()[i];
                    rem /= C.dims()[i];
                }
                size_t flatA = 0, flatB = 0;
                for (size_t i = 0; i < N; ++i) {
                    size_t ia = (A.dims()[i] == 1) ? 0 : idxC[i];
                    size_t ib = (B.dims()[i] == 1) ? 0 : idxC[i];
                    flatA += ia * A.strides()[i];
                    flatB += ib * B.strides()[i];
                }
                C.data()[lin] = op(A.data()[flatA], B.data()[flatB]);
            }
            return C;
        }

        // Operadores binarios usando la función genérica
        template <typename T, size_t N>
        Tensor<T, N> operator+(const Tensor<T, N>& A, const Tensor<T, N>& B) {
            return _binary_op(A, B, std::plus<T>());
        }

        template <typename T, size_t N>
        Tensor<T, N> operator-(const Tensor<T, N>& A, const Tensor<T, N>& B) {
            return _binary_op(A, B, std::minus<T>());
        }

        template <typename T, size_t N>
        Tensor<T, N> operator*(const Tensor<T, N>& A, const Tensor<T, N>& B) {
            return _binary_op(A, B, std::multiplies<T>());
        }

// Escalares
        template <typename T, size_t N> Tensor<T,N> operator+(const Tensor<T,N>& A, const T& s) { Tensor<T,N> C=A; for (auto& v : C.data()) v += s; return C; }
        template <typename T, size_t N> Tensor<T,N> operator+(const T& s, const Tensor<T,N>& A) { return A + s; }
        template <typename T, size_t N> Tensor<T,N> operator-(const Tensor<T,N>& A, const T& s) { Tensor<T,N> C=A; for (auto& v : C.data()) v -= s; return C; }
        template <typename T, size_t N> Tensor<T,N> operator*(const Tensor<T,N>& A, const T& s) { Tensor<T,N> C=A; for (auto& v : C.data()) v *= s; return C; }
        template <typename T, size_t N> Tensor<T,N> operator*(const T& s, const Tensor<T,N>& A) { return A * s; }
        template <typename T, size_t N> Tensor<T,N> operator/(const Tensor<T,N>& A, const T& s) {
            if (s == T{}) throw std::invalid_argument("Division by zero");
            Tensor<T,N> C=A; for (auto& v : C.data()) v /= s; return C;
        }

// Transpuesta (solo para tensores 2D+)
        template <typename T, size_t N>
        Tensor<T,N> transpose_2d(const Tensor<T,N>& t) {
            if constexpr (N < 2) throw std::invalid_argument("Cannot transpose 1D tensor: need at least 2 dimensions"
);

            std::array<size_t, N> new_dims = t.dims();
            std::swap(new_dims[N-2], new_dims[N-1]);
            Tensor<T,N> R(new_dims);

            for (size_t i = 0; i < R.size(); ++i) {
                std::array<size_t, N> idxR, idxT;
                size_t index = i;
                for (int j = N-1; j >= 0; --j) {
                    idxR[j] = index % R.dims()[j];
                    index /= R.dims()[j];
                }
                idxT = idxR;
                std::swap(idxT[N-2], idxT[N-1]);

                size_t linT = 0;
                for (size_t j = 0; j < N; ++j) linT += idxT[j] * t.strides()[j];
                R.data()[i] = t.data()[linT];
            }

            return R;
        }


        template <typename T>
        Tensor<T, 2> matrix_product(const Tensor<T, 2>& A, const Tensor<T, 2>& B) {
            if (A.shape()[1] != B.shape()[0])
                throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");

            size_t M = A.shape()[0];
            size_t K = A.shape()[1];
            size_t P = B.shape()[1];

            Tensor<T, 2> R(std::array<size_t, 2>{M, P});

            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < P; ++j) {
                    T sum = T();
                    for (size_t k = 0; k < K; ++k)
                        sum += A(i, k) * B(k, j);
                    R(i, j) = sum;
                }
            }

            return R;
        }

        // Producto matricial (soporta batch)
        template <typename T, size_t N>
        Tensor<T,N> matrix_product(const Tensor<T,N>& A, const Tensor<T,N>& B) {
            static_assert(N >= 2, "Number of dimensions do not match with 2");

            for (size_t i = 0; i < N - 2; ++i) {
                if (A.dims()[i] != B.dims()[i])
                    throw std::invalid_argument("Matrix dimensions are compatible for multiplication BUT Batch dimensions do not match");
            }

            if (A.dims()[N-1] != B.dims()[N-2])
                throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");

            std::array<size_t, N> result_dims;
            for (size_t i = 0; i < N - 2; ++i) result_dims[i] = A.dims()[i];
            result_dims[N-2] = A.dims()[N-2];
            result_dims[N-1] = B.dims()[N-1];

            Tensor<T,N> R(result_dims);

            size_t batch = 1;
            for (size_t i = 0; i < N - 2; ++i) batch *= A.dims()[i];

            size_t M = A.dims()[N-2], K = A.dims()[N-1], P = B.dims()[N-1];

            for (size_t b = 0; b < batch; ++b) {
                size_t offsetA = b * M * K;
                size_t offsetB = b * K * P;
                size_t offsetR = b * M * P;

                for (size_t i = 0; i < M; ++i)
                    for (size_t j = 0; j < P; ++j) {
                        T sum{};
                        for (size_t k = 0; k < K; ++k)
                            sum += A.data()[offsetA + i*K + k] * B.data()[offsetB + k*P + j];
                        R.data()[offsetR + i*P + j] = sum;
                    }
            }

            return R;
        }

        template<typename T, size_t N, typename Func>
Tensor<T, N> apply(const Tensor<T, N>& t, Func f) {
            Tensor<T, N> result = t;
            for (T& val : result.data()) {
                val = f(val);
            }
            return result;
        }

    } // namespace algebra
} // namespace utec

#endif //PROG3_TENSOR_FINAL_PROJECT_V2025_01_TENSOR_H
