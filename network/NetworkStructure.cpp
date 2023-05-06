#include "NetworkStructure.h"
#include <random>

using namespace Network;

Network::NeuronLayer::NeuronLayer(int neuronCount, int prevCount)
{
	this->neuronCount = neuronCount;
	this->prevCount = prevCount;

	value = new float_n[neuronCount];
	error = new float_n[neuronCount];

	ClearValues();

	bias = 0.0;

	for (int i = 0; i < neuronCount; i++)
		weightList.push_back(new float_n[prevCount]);
}

NeuronLayer::NeuronLayer()
{
	// initialize parameters
	bias = 0.0;
	prevCount = 0;
	neuronCount = 0;

	value = nullptr;
	error = nullptr;
}

void NeuronLayer::InitAllWeights(float_n weight)
{
	for (int neuron = 0; neuron < neuronCount; neuron++)
		for (int i = 0; i < prevCount; i++)
			weightList[neuron][i] = weight;
}

void NeuronLayer::RandomizeWeightAndBias(float_n min, float_n max)
{
	std::mt19937 rnd(std::random_device{}());
	std::uniform_real_distribution<float_n> dist(min, max);

	for (int neuron = 0; neuron < neuronCount; neuron++)
	{
		for (int i = 0; i < prevCount; i++)
		{
			weightList[neuron][i] = dist(rnd);
		}
	}

	bias = dist(rnd);
}

NeuronLayerInstance::NeuronLayerInstance(NeuronLayer* source)
{
	this->source = source;

	prevCount = source->prevCount;
	neuronCount = source->neuronCount;
	bias = source->bias;

	value = new float_n[neuronCount];
	error = new float_n[neuronCount];
}

void NeuronLayerInstance::FeedBack()
{
	for (int i = 0; i < neuronCount; i++)
	{
		source->value[i] += value[i];
		source->error[i] += error[i];
	}
}

void Network::NeuronLayerInstance::Free()
{
	delete[] value;
	delete[] error;
}

void NeuronLayerInstance::ClearSourceValue()
{
	source->ClearValues();
}

void NeuronLayerInstance::FetchBias()
{
	bias = source->bias;
}

float_n* Network::NeuronLayerInstance::operator[](int index)
{
	return source->weightList[index];
}

void NeuronLayerInstance::PushDataFloat(float_n* data)
{
	memcpy(value, data, sizeof(float_n) * neuronCount);
}

void NeuronLayer::ClearValues()
{
	for (int i = 0; i < neuronCount; i++)
	{
		value[i] = 0.0;
		error[i] = 0.0;
	}
}

float_n* Network::NeuronLayer::operator[](int index)
{
	return weightList[index];
}

void Network::NeuronLayer::Free()
{
	for (auto neuron : weightList)
		delete[] neuron;

	weightList.clear();

	delete[] value;
	delete[] error;
}
