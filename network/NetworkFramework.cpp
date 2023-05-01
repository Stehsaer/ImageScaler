#include "NetworkFramework.h"

#include "NetworkStructure.h"
#include "NetworkAlgorithm.h"
#include "NetworkData.h"

#include <float.h>

using namespace Network::Connectivity;
using namespace Network::Algorithm;

typedef Network::float_n float_n;

FullConnNetwork::FullConnNetwork(int inNeuronCount, int outNeuronCount, int hiddenNeuronCount, int hiddenLayerCount, ActivateFunctionType ActivateFunc, float_n learningRate, bool outLayerSoftMax)
{
	if (inNeuronCount <= 0 || outNeuronCount <= 0 || hiddenNeuronCount <= 0 || hiddenLayerCount <= 0 || learningRate < 0.0)
	{
		throw std::exception("Invalid Parameters");
	}

	// Initialize parameters
	this->inNeuronCount = inNeuronCount;
	this->outNeuronCount = outNeuronCount;
	this->hiddenNeuronCount = hiddenNeuronCount;
	this->hiddenLayerCount = hiddenLayerCount;
	this->learningRate = learningRate;
	this->loss = 0.0;
	this->targetData = new float_n[outNeuronCount];

	this->ActivateFunc = ActivateFunc;

	ForwardActive = forwardFuncList[(int)ActivateFunc];
	BackwardActive = backwardFuncList[(int)ActivateFunc];

	inLayer = NeuronLayer(inNeuronCount, 0);
	outLayer = NeuronLayer(outNeuronCount, hiddenNeuronCount);

	this->outLayerSoftMax = outLayerSoftMax;
	
	// add layers
	for (int i = 0; i < hiddenLayerCount; i++)
	{
		hiddenLayerList.push_back(NeuronLayer(hiddenNeuronCount, i == 0 ? inNeuronCount : hiddenNeuronCount));
	}
}

FullConnNetwork::FullConnNetwork(int inNeuronCount, NeuronLayer outLayer, int hiddenNeuronCount, int hiddenLayerCount, ActivateFunctionType activateFunc, float_n learningRate, bool outLayerSoftMax)
{
	if (inNeuronCount <= 0 || outLayer.neuronCount == 0)
	{
		throw std::exception("Invalid Parameters");
	}

	this->inNeuronCount = inNeuronCount;
	outNeuronCount = outLayer.neuronCount;

	this->hiddenNeuronCount = hiddenNeuronCount;
	this->hiddenLayerCount = hiddenLayerCount;

	ActivateFunc = activateFunc;
	this->learningRate = learningRate;

	this->loss = 0.0;
	this->targetData = new float_n[outNeuronCount];

	ForwardActive = forwardFuncList[(int)ActivateFunc];
	BackwardActive = backwardFuncList[(int)ActivateFunc];

	inLayer = NeuronLayer(inNeuronCount, 0);

	this->outLayer = outLayer;
	this->outLayerSoftMax = outLayerSoftMax;
}

float_n FullConnNetwork::GetLoss()
{
	loss = 0.0;
	
	for (int i = 0; i < outNeuronCount; i++)
	{
		loss += (outLayer.value[i] - targetData[i]) * (outLayer.value[i] - targetData[i]);
	}

	return loss;
}

void FullConnNetwork::RandomizeAllWeights(float_n min, float_n max)
{
	outLayer.RandomizeWeightAndBias(min, max);

	for (auto& layer : hiddenLayerList)
	{
		layer.RandomizeWeightAndBias(min, max);
	}
}

void FullConnNetwork::SetAllWeights(float_n weight)
{
	inLayer.InitAllWeights(weight);
	outLayer.InitAllWeights(weight);

	for (auto& layer : hiddenLayerList)
	{
		layer.InitAllWeights(weight);
	}
}

void FullConnNetwork::PushDataDouble(double* data)
{
	for (int i = 0; i < inNeuronCount; i++)
	{
		inLayer.value[i] = data[i];
	}
}

void Network::Connectivity::FullConnNetwork::PushDataFloat(float_n* data)
{
	memcpy(inLayer.value, data, sizeof(float) * inLayer.neuronCount);
}

void FullConnNetwork::ForwardTransmitLayer(NeuronLayer& obj, NeuronLayer& prev) 
{
	for (int i = 0; i < obj.neuronCount; i++) 
	{
		float sum = 0.0;

		float_n* weights = obj[i];
		float_n* value = prev.value;

#pragma omp simd reduction(+:sum)
		for (int j = 0; j < obj.prevCount; j++)
		{
			sum += value[j] * weights[j];
		}

		obj.value[i] = sum + obj.bias;
		obj.value[i] = (*ForwardActive)(obj.value[i] / (float_n)obj.prevCount);
	}
}

void FullConnNetwork::ForwardTransmitLayer(NeuronLayerInstance& obj, NeuronLayerInstance& prev)
{
	for (int i = 0; i < obj.neuronCount; i++)
	{
		obj.value[i] = 0.0;

		float_n* weights = obj[i];

		for (int j = 0; j < obj.prevCount; j++)
		{
			obj.value[i] += prev.value[j] * weights[j];
		}

		obj.value[i] += obj.bias;
		obj.value[i] = (*ForwardActive)(obj.value[i] / (float_n)obj.prevCount);
	}
}

void FullConnNetwork::ForwardTransmit()
{
	for (int i = 0; i < hiddenLayerCount; i++)
	{
		ForwardTransmitLayer(hiddenLayerList[i], i == 0 ? inLayer : hiddenLayerList[i - 1]);
	}

	NeuronLayer& lastHiddenLayer = hiddenLayerList[hiddenLayerCount - 1];

	for (int i = 0; i < outLayer.neuronCount; i++)
	{
		outLayer.value[i] = 0.0;

		float_n* weights = outLayer[i];

		for (int j = 0; j < outLayer.prevCount; j++)
		{
			outLayer.value[i] += lastHiddenLayer.value[j] * weights[j];
		}

		outLayer.value[i] += outLayer.bias;
		outLayer.value[i] = outLayer.value[i] / outLayer.prevCount;
	}

	if(outLayerSoftMax) SoftMax(outLayer);
}

void FullConnNetwork::BackwardTransmitLayer(NeuronLayer& obj, NeuronLayer& last)
{
	for (int i = 0; i < obj.neuronCount; i++)
	{
		obj.error[i] = 0.0;

		for (int j = 0; j < last.neuronCount; j++)
		{
			obj.error[i] += last.error[j] * last[j][i];
		}
	}
}

void FullConnNetwork::BackwardTransmitLayer(NeuronLayerInstance& obj, NeuronLayerInstance& last)
{
	for (int i = 0; i < obj.neuronCount; i++)
	{
		obj.error[i] = 0.0;

		for (int j = 0; j < last.neuronCount; j++)
		{
			obj.error[i] += last.error[j] * last[j][i];
		}
	}
}

void FullConnNetwork::BackwardTransmit()
{
	if (outLayerSoftMax) SoftMaxGetError(outLayer, targetData);
	else
	{
		for (int i = 0; i < outLayer.neuronCount; i++)
		{
			outLayer.error[i] = targetData[i] - outLayer.value[i];
		}
	}

	for (int i = hiddenLayerCount - 1; i >= 0; i--)
	{
		BackwardTransmitLayer(hiddenLayerList[i], i == hiddenLayerCount - 1 ? outLayer : hiddenLayerList[i + 1]);
	}
}

void FullConnNetwork::UpdateLayerWeights(NeuronLayer& layer, NeuronLayer& lastLayer)
{
	for (int i = 0; i < layer.neuronCount; i++)
	{
		float_n coeff = learningRate * (*BackwardActive)(layer.value[i]) * layer.error[i]; // common coeff

		layer.bias += learningRate * (*BackwardActive)(layer.bias) * layer.error[i]; // tweak bias

		for (int j = 0; j < layer.prevCount; j++)
		{
			layer[i][j] += coeff * lastLayer.value[j];
		}
	}
}

void FullConnNetwork::computeAverage(float_n count)
{
	for (auto& layer : hiddenLayerList)
	{
		for (int i = 0; i < layer.neuronCount; i++)
		{
			layer.value[i] /= count;
			layer.error[i] /= count;
		}
	}
	for (int i = 0; i < inLayer.neuronCount; i++)
	{
		inLayer.value[i] /= count;
		inLayer.error[i] /= count;
	}
	for (int i = 0; i < outLayer.neuronCount; i++)
	{
		outLayer.value[i] /= count;
		outLayer.error[i] /= count;
	}
}

void FullConnNetwork::UpdateWeights()
{
	UpdateLayerWeights(outLayer, hiddenLayerList[hiddenLayerCount - 1]); // update outlayer

	for (int i = 0; i < hiddenLayerCount; i++)
	{
		UpdateLayerWeights(hiddenLayerList[i], i == 0 ? inLayer : hiddenLayerList[i - 1]);
	}
}

int FullConnNetwork::FindLargestOutput()
{
	float_n biggest = outLayer.value[0];
	int biggestNeuron = 0;

	for (int i = 1; i < outLayer.neuronCount; i++)
	{
		if (outLayer.value[i] > biggest)
		{
			biggest = outLayer.value[i];
			biggestNeuron = i;
		}
	}

	return biggestNeuron;
}

int FullConnNetwork::GetResultDouble(double* data)
{
	PushDataDouble(data);
	ForwardTransmit();
	return FindLargestOutput();
}

int FullConnNetwork::GetResultFloat(float* data)
{
	PushDataFloat(data);
	ForwardTransmit();
	return FindLargestOutput();
}

int FullConnNetwork::GetResultNetworkData(NetworkData& data)
{
	return GetResultFloat(data.data);
}

float_n FullConnNetwork::GetAccuracy(NetworkDataSet& set)
{
	int correctCount = 0;

	for (auto& data : set.dataSet)
	{
		int label_out = GetResultNetworkData(*data);
		if (label_out == data->label)
		{
			correctCount++;
		}
	}

	return (float_n)correctCount / (float_n)set.Count();
}

float_n FullConnNetwork::GetAccuracyCallbackFloat(NetworkDataSet& set, float* progressVariable)
{
	int correctCount = 0, doneCount = 0;

	for (auto& data : set.dataSet)
	{
		int label_out = GetResultNetworkData(*data);
		if (label_out == data->label)
		{
			correctCount++;
		}

		doneCount++;

		*progressVariable = (float)doneCount / (float)set.Count();
	}

	return (float_n)correctCount / (float_n)set.Count();
}

void FullConnNetwork::PushTargetLabel(int label)
{
	for (int i = 0; i < outNeuronCount; i++)
	{
		targetData[i] = i == label ? 1.0 : 0.0;
	}
}

void Network::Connectivity::FullConnNetwork::PushTargetFloat(float_n* data)
{
	memcpy(targetData, data, outNeuronCount * sizeof(float_n));
}

void FullConnNetwork::Destroy()
{
	// Clear neuron data
	inLayer.Free();

	outLayer.Free();

	for (NeuronLayer& layer : hiddenLayerList)
	{
		layer.Free();
	}

	hiddenLayerList.clear();

	// Clear target data
	if (targetData) delete[] targetData;
}
