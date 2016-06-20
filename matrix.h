#pragma once

#include "aligned_allocator.h"

#include <vector>
#include <cstddef>
#include <stdexcept>

template <typename T>
class Matrix {
    using vector_type = std::vector<T, AlignedAllocator<T>>;
    public:
        Matrix(): Matrix(0, 0) { }
        Matrix(size_t rows, size_t cols): Matrix(rows, cols, T()) { }
        Matrix(size_t rows, size_t cols, const T& fill)
            : numRows_(rows), numCols_(cols), data_(numRows_ * numCols_, fill) { }

        const T& operator()(size_t row, size_t col) const {
            return data_[row*numCols_ + col];
        }

        T& operator()(size_t row, size_t col) {
            return data_[row*numCols_ + col];
        }

        const T& at(size_t row, size_t col) const {
            if (!(between(row, 0, numRows_) && between(col, 0, numCols_))) {
                throw std::runtime_error("Invalid row/column\n");
            }
            return operator()(row, col);
        }

        T& at(size_t row, size_t col) {
            if (!(between(row, 0, numRows_) && between(col, 0, numCols_))) {
                throw std::runtime_error("Invalid row/column\n");
            }

            return operator()(row, col);
        }

        using iterator = typename vector_type::iterator;
        using const_iterator = typename vector_type::const_iterator;

        iterator begin() {
            return data_.begin();
        }

        const_iterator begin() const {
            return data_.begin();
        }

        const_iterator cbegin() const {
            return begin();
        }

        iterator end() {
            return data_.end();
        }

        const_iterator end() const {
            return data_.end();
        }

        const_iterator cend() const {
            return end();
        }

        size_t getRows() const {
            return numRows_;
        }

        size_t getCols() const {
            return numCols_;
        }

    private:
        size_t numRows_, numCols_;
        vector_type data_;

        template <typename U>
        bool between(U value, U min, U max) {
            return min <= value && value <= max;
        }
};
