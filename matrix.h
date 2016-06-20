#pragma once

#include "aligned_allocator.h"

#include <vector>
#include <cstddef>
#include <stdexcept>

template <typename T>
class Matrix {
	using vector_type = std::vector<T, AlignedAllocator<T>>;
public:
	using value_type = T;
	using reference = T&;
	using const_reference = const T&;
	using pointer = T*;
	using const_pointer = const T*;
	using difference_type = std::ptrdiff_t;
	using size_type = std::size_t;

	Matrix(): Matrix(0, 0) { }
	Matrix(std::size_t rows, std::size_t cols): Matrix(rows, cols, T()) { }
	Matrix(std::size_t rows, std::size_t cols, const_reference fill)
		: numRows_(rows), numCols_(cols), data_(numRows_ * numCols_, fill) { }

	const_reference operator()(std::size_t row, std::size_t col) const {
		return data_[row*numCols_ + col];
	}

	reference operator()(std::size_t row, std::size_t col) {
		return data_[row*numCols_ + col];
	}

	const_reference at(std::size_t row, std::size_t col) const {
		if (!(between(row, 0, numRows_) && between(col, 0, numCols_))) {
			throw std::runtime_error("Invalid row/column\n");
		}
		return operator()(row, col);
	}

	const_reference at(std::size_t row, std::size_t col) {
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

	std::size_t getRows() const {
		return numRows_;
	}

	std::size_t getCols() const {
		return numCols_;
	}

private:
	std::size_t numRows_, numCols_;
	vector_type data_;

	template <typename U>
	bool between(U value, U min, U max) {
		return min <= value && value < max;
	}
};
