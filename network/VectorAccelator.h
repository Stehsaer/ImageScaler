#pragma once

// Accelerate vector computations by utilizing SIMD instructions
class VectorAccelator
{
public:
	/// <summary>
	/// Dot Product for 32-bit float
	/// </summary>
	/// <returns>Dot product</returns>
	static float Dot(const float* a, const float* b, unsigned int count);
};
