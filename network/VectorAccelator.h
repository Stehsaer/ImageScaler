#pragma once

#ifdef __AVX2__
#include <immintrin.h>
#endif

// Accelerate vector computations by utilizing SIMD instructions
class VectorAccelator
{
public:

	// Dot Product for 32-bit float
	static inline float Dot(const float* a, const float* b, unsigned int count)
	{
#ifdef __AVX2__

		int j; __m256 sum = _mm256_set_ps(0, 0, 0, 0, 0, 0, 0, 0);

		for (j = 0; j < count; j += 8)
		{
			__m256 a_256 = _mm256_loadu_ps(a + j);
			__m256 b_256 = _mm256_loadu_ps(b + j);
			__m256 result = _mm256_mul_ps(a_256, b_256);
			sum = _mm256_add_ps(sum, result);
		}

		float sum_f; float* sum_ptr = (float*)&sum;
		sum_f = sum_ptr[0] + sum_ptr[1] + sum_ptr[2] + sum_ptr[3] + sum_ptr[4] + sum_ptr[5] + sum_ptr[6] + sum_ptr[7]; // add up results in 256-bit packed result

		// add rest
		for (; j < count; j++)
		{
			sum_f += a[j] * b[j];
		}

		return sum_f;

#else
		float sum = 0;

		for (unsigned int i = 0; i < count; i++)
		{
			sum += a[i] * b[i];
		}

		return sum;
#endif
	}
};

