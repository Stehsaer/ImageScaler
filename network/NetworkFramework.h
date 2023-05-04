#ifndef _NETWORK_FRAMEWORK_H_
#define _NETWORK_FRAMEWORK_H_

#include <vector>
#include <functional>
#include <atomic>

#include "NetworkStructure.h"
#include "NetworkData.h"
#include "../Image.h"

namespace Network
{
	namespace Connectivity
	{
		class FullConnNetwork
		{
		public:
			// active functions
			ActivateFunction ForwardActive;
			ActivateFunction BackwardActive;

			ActivateFunctionType ActivateFunc;

			// neuron counts
			int inNeuronCount, outNeuronCount, hiddenNeuronCount, hiddenLayerCount;

			// layers
			NeuronLayer inLayer, outLayer;
			std::vector<NeuronLayer> hiddenLayerList;

			bool outLayerSoftMax;

			// training parameters
			float_n learningRate, loss;
			float_n* targetData;

			FullConnNetwork(int inNeuronCount, int outNeuronCount, int hiddenNeuronCount, int hiddenLayerCount, ActivateFunctionType activateFunc, float_n learningRate = 0.0, bool outLayerSoftMax = true);

			FullConnNetwork(int inNeuronCount, NeuronLayer outLayer, int hiddenNeuronCount, int hiddenLayerCount, ActivateFunctionType activateFunc, float_n learningRate = 0.0, bool outLayerSoftMax = true);

			float_n GetLoss();

			void RandomizeAllWeights(float_n min, float_n max);

			void SetAllWeights(float_n weight);

			void PushDataDouble(double* data);
			void PushDataFloat(float_n* data);
			void PushTargetLabel(int label);
			void PushTargetFloat(float_n* data);

			void ForwardTransmit();
			void ForwardTransmit(NeuronLayerInstance& inLayer, NeuronLayerInstance& outLayer, std::vector<NeuronLayerInstance>& hiddenLayers);

			void BackwardTransmit();

			void UpdateWeights();

			int FindLargestOutput();

			int GetResultNetworkData(NetworkData& data);
			int GetResultDouble(double* data);
			int GetResultFloat(float* data);

			float_n GetAccuracy(NetworkDataSet& set);
			float_n GetAccuracyCallbackFloat(NetworkDataSet& set, float* progressVariable);

			void Destroy();

			// Train Funcitons

			void ForwardTransmitLayer(NeuronLayer& obj, NeuronLayer& prev);
			void BackwardTransmitLayer(NeuronLayer& obj, NeuronLayer& last);
			void UpdateLayerWeights(NeuronLayer& layer, NeuronLayer& lastLayer);

			void ForwardTransmitLayer(NeuronLayerInstance& obj, NeuronLayerInstance& prev);
			void BackwardTransmitLayer(NeuronLayerInstance& obj, NeuronLayerInstance& last);

			void computeAverage(float_n count);

			void TrainBatched(NetworkDataSet& dataset, int batchSize, float_n learningRate, std::function<void(int, int)> callback = [](int, int) {});
			void TrainBatched(std::vector<ImageDataset*>& dataset, int batchSize, float_n learningRate, std::function<void(int, int)> callback = [](int, int) {});
		};

		class FullConnNetworkInstance
		{
		public:
			// variables

			FullConnNetwork* source;

			NeuronLayerInstance inLayer;
			NeuronLayerInstance outLayer;
			std::vector<NeuronLayerInstance> hiddenLayerList;

			float_n* target;

			FullConnNetworkInstance(FullConnNetwork* src);
			
			// Data Management

			void PushData(float_n* data);
			void PushTarget(float_n* target);
			void FreeData();

			// Transmission

			void ForwardTransmit();
			void BackwardTransmit();
			void FeedBack();
			void FetchBias();

			float_n GetLoss();
		};
	}
}

#endif
