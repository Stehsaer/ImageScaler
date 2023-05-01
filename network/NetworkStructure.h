#ifndef _NETWORK_STRUCTURE_H_
#define _NETWORK_STRUCTURE_H_

#include <vector>
#include <atomic>

namespace Network
{
	typedef float float_n;

	class NeuronLayer
	{
	public:
		std::vector<float_n*> weightList;
		float_n* value;
		float_n* error;
		int neuronCount;
		int prevCount; // count of neurons of previous layer

		float_n bias;

		NeuronLayer(int neuronCount, int prevCount);
		NeuronLayer();

		void InitAllWeights(float_n weight);
		void RandomizeWeightAndBias(float_n min, float_n max);
		void ClearValues();

		float_n* operator[](int index);

		void Free();
	};

	// Used for batched gradient descent, doesn't affect original NeuronLayer
	class NeuronLayerInstance
	{
	public:
		NeuronLayer* source;

		float_n* value;
		float_n* error;

		int prevCount;
		int neuronCount;

		float_n bias;

		NeuronLayerInstance(NeuronLayer* source);

		void FeedBack();

		void Free();
		void ClearSourceValue();
		void FetchBias();

		float_n* operator[](int index);

		void PushDataFloat(float_n* data);
	};
}

#endif