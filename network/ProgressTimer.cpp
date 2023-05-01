#include "ProgressTimer.h"

ProgressTimer::ProgressTimer()
{
	Reset();
}

void ProgressTimer::Reset()
{
	startTimePoint = std::chrono::steady_clock::now();
}

long long ProgressTimer::Count()
{
	return (std::chrono::steady_clock::now() - startTimePoint).count();
}

long long ProgressTimer::CountMs()
{
	long long duration = Count();
	
	return duration / 1000000LL;
}

long long ProgressTimer::CountAndReset()
{
	long long temp = Count();
	Reset();
	return temp;
}