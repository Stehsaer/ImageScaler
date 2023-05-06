#include "VectorAccelator.h"

float VectorAccelator::Dot(const float* a, const float* b, unsigned int count)
{
	float sum = 0;

	for (unsigned int i = 0; i < count; i++)
	{
		sum += a[i] * b[i];
	}

	return sum;
}
