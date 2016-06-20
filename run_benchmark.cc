#include "matrix.h"

#include <array>
#include <vector>
#include <functional>
#include <string>
#include <utility>
#include <tuple>
#include <chrono>
#include <random>

#include <cstddef>
#include <cstdio>

#if defined(_MSC_VER)
#pragma warning (disable : 4127, 4244)
#endif

#include "vectorclass.h"

#if defined(_MSC_VER)
#pragma warning (4 : 4127, 4244)
#endif

#if defined(_MSC_VER)
#pragma warning (disable: 4996)
#endif

using std::vector;
using std::array;
using std::function;
using std::string;
using std::tuple;
using std::make_tuple;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::default_random_engine;
using std::uniform_real_distribution;

#define MAX_SIZE 10000
#define MIN_SIZE 100
#define NUM_SIZES 10
#define NUM_TRIALS 10

template <typename NumberType>
void simple_matrix_multiply(const Matrix<NumberType>& left, const Matrix<NumberType>& right, Matrix<NumberType>* outPtr) {
	Matrix<NumberType>& out = *outPtr;

	const size_t numRows = left.getRows();
	const size_t numCols = left.getCols();
	for (size_t i = 0; i < numRows; ++i) {
		for (size_t j = 0; j < numCols; ++j) {
			double result = 0;
			for (size_t k = 0; k < numRows; ++k) {
				result += left(i, k) * right(k, j);
			}
			out(i, j) = result;
		}
	}
}

template <typename NumberType>
void improved_matrix_multiply(const Matrix<NumberType>& left, const Matrix<NumberType>& right, Matrix<NumberType>* outPtr) {
	Matrix<NumberType>& out = *outPtr;

	const size_t numRows = left.getRows();
	const size_t numCols = left.getCols();

	for (size_t i = 0; i < numRows; ++i) {
		for (size_t j = 0; j < numCols; ++j) {
			double result = 0;
			Vec4d result_simd(0);
			size_t k = 0;

			for (; k < (numRows & ~3); k += 4) {
				Vec4d left_simd;
				left_simd.load_a(&left(i, k));
				Vec4d right_simd(right(j, k), right(j, k + 1), right(j, k + 2), right(j, k + 3));
				result_simd += left_simd * right_simd;
			}
			for (; k < numRows; ++k) {
				result += left(i, k) * right(k, j);
			}

			out(i, j) = result + horizontal_add(result_simd);
		}
	}
}

#ifdef OMP_ENABLED
template <typename NumberType>
void omp_matrix_multiply(const Matrix<NumberType>& left, const Matrix<NumberType>& right, Matrix<NumberType>* outPtr) {
	Matrix<NumberType>& out = *outPtr;

	const long numRows = left.getRows();
	const long numCols = left.getCols();
	for (long i = 0; i < numRows; ++i) {
#pragma omp parallel for default(none), private(i), shared(left, right, out)
		for (long j = 0; j < numCols; ++j) {
			double result = 0;
			for (long k = 0; k < numRows; ++k) {
				result += left(i, k) * right(k, j);
			}
			out(i, j) = result;
		}
	}
}
#endif

#ifdef CUDA_ENABLED
#include "matrix.cuh"
#endif

#ifdef OPENCL_ENABLED
#include "matrix.cl"
#endif

#ifdef KOKKOS_ENABLED
#include "matrix_kokkos.h"
#endif

template <typename NumberType>
using MatrixMultiplyFunction = function<
	void(const Matrix<NumberType>&,
			const Matrix<NumberType>&,
			Matrix<NumberType>*)>;

vector<tuple<MatrixMultiplyFunction<double>, string, FILE*>> functions = {
	make_tuple(static_cast<MatrixMultiplyFunction<double>>(simple_matrix_multiply<double>), "simple", nullptr),
	make_tuple(static_cast<MatrixMultiplyFunction<double>>(improved_matrix_multiply<double>), "improved", nullptr),
#ifdef CUDA_ENABLED
	make_tuple(static_cast<MatrixMultiplyFunction<double>>(cude_matrix_multiply<double>), "cuda", nullptr),
#endif
#ifdef OPENCL_ENABLED
	make_tuple(static_cast<MatrixMultiplyFunction<double>>(opencl_matrix_multiply<double>), "opencl", nullptr),
#endif
#ifdef OMP_ENABLED
	make_tuple(static_cast<MatrixMultiplyFunction<double>>(omp_matrix_multiply<double>), "omp", nullptr),
#endif
#ifdef KOKKOS_OMP_ENABLED
	make_tuple(static_cast<MatrixMultiplyFunction<double>>(kokkos_omp_matrix_multiply<double>), "kokkos_omp", nullptr),
#endif
#ifdef KOKKOS_CUDA_ENABLED
	make_tuple(static_cast<MatrixMultiplyFunction<double>>(kokkos_cuda_matrix_multiply<double>), "kokkos_cuda", nullptr),
#endif
#ifdef KOKKOS_OPENCL_ENABLED
	make_tuple(static_cast<MatrixMultiplyFunction<double>>(kokkos_opencl_matrix_multiply<double>), "kokkos_opencl", nullptr),
#endif
};

template <typename NumberType>
void check_result(const Matrix<NumberType>& expected, const Matrix<NumberType>& actual) {
	if (expected.getRows() != actual.getRows() || expected.getCols() != actual.getCols()) {
		throw std::runtime_error("Matrices are not the same size!\n");
	}

	auto it_expected = expected.begin();
	auto it_actual = actual.begin();
	for (; it_expected != expected.begin(); ++it_actual, ++it_expected) {
		if (*it_expected != *it_actual) {
			char sprintf_buffer[100];
			sprintf(sprintf_buffer, "Values not the same: expected %.3f, got %.3f\n", *it_expected, *it_actual);
			throw std::runtime_error(sprintf_buffer);
		}
	}
}

template <typename MatrixType, typename FunctionType>
double timing_test(
		tuple<FunctionType, string, FILE*> functions,
		size_t num_attempts, 
		const MatrixType& left, 
		const MatrixType& right, 
		const MatrixType& correct_values) {

	double result = std::numeric_limits<double>::max();

	for (size_t attempt = 0; attempt < num_attempts; ++attempt) {
		MatrixType resultMatrix(left.getRows(), left.getCols(), 0);

		auto tic = high_resolution_clock::now();
		std::get<0>(functions)(left, right, &resultMatrix);
		auto toc = high_resolution_clock::now();

		check_result(correct_values, resultMatrix);

		const double time_elapsed = duration_cast<duration<double>>(toc - tic).count();

		result = std::min(result, time_elapsed);
	}

	return result;
}

int main() {
	default_random_engine engine;

	char sprintf_buffer[50];
	for (auto& item : functions) {
		sprintf(sprintf_buffer, "%s_results.csv", std::get<1>(item).c_str());
		std::get<2>(item) = fopen(sprintf_buffer, "w");
	}

	const size_t difference = (MAX_SIZE - MIN_SIZE) / NUM_SIZES;

	for (size_t size = MIN_SIZE; size < MAX_SIZE; size += difference) {
		fprintf(stderr, "Starting size %zu...\n", size);
		uniform_real_distribution<double> gen(0, 10);

		Matrix<double> left(size, size, 0.);
		Matrix<double> right(size, size, 0.);

		auto it_left = left.begin();
		auto it_right = right.begin();
		for (; it_left != left.end(); ++it_left, ++it_right) {
			*it_left = gen(engine);
			*it_right = gen(engine);
		}

		Matrix<double> correct_result(size, size, 0.);
		std::get<0>(functions[0])(left, right, &correct_result);

		auto tic = high_resolution_clock::now();

		for (const auto& items : functions) {
			fprintf(stderr, "\tStarting version %s...", std::get<1>(items).c_str());
			auto innertic = high_resolution_clock::now();
			double best_time = timing_test(items, NUM_TRIALS, left, right, correct_result);
			auto innertoc = high_resolution_clock::now();
			const double time = duration_cast<duration<double>>(innertoc - innertic).count();
			fprintf(stderr, "Finished in %.3f seconds (best %.3f)\n", time, best_time);
			fprintf(std::get<2>(items), ",%.3f", best_time);
		}

		auto toc = high_resolution_clock::now();

		const double time = duration_cast<duration<double>>(toc - tic).count();
		fprintf(stderr, "Finished size %zu in %.3f seconds\n", size, time);
	}

	for (const auto& item : functions) {
		fclose(std::get<2>(item));
	}
}
