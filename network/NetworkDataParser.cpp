#include "NetworkDataParser.h"
#include "ProgressTimer.h"
#include "FileHelper.h"

#include <format>
#include <json/json.h>

using namespace Network;
using FCNetwork = Network::Connectivity::FullConnNetwork;

/// <summary>
/// Tell if the computer runs on a little-endian system
/// </summary>
/// <returns>True for little-endian, False for big-endian</returns>
bool isLittleEndian()
{
	union U
	{
		int  i;
		char c;
	};
	U u;
	u.i = 1;
	return u.c == 1;
}

/// <summary>
/// Convert a number between MSB and LSB
/// </summary>
/// <param name="x">Target int</param>
void Reverse(int* x)
{
	int y = (0xff000000 & *x) >> 24;
	y = y | (0x00ff0000 & *x) >> 8;
	y = y | (0x0000ff00 & *x) << 8;
	y = y | (0x000000ff & *x) << 24;
	*x = y;
}

/// <summary>
/// Get int number from a specfic memory location
/// </summary>
/// <param name="data">Data pointer</param>
/// <param name="offset">Offset from the starting position</param>
/// <returns>Int</returns>
int GetInt32NoReverse(unsigned char* data, int offset)
{
	int out;
	memcpy(&out, data + offset, sizeof(int));
	return out;
}

/// <summary>
/// Get int number from a specfic memory location
/// </summary>
/// <param name="data">Data pointer</param>
/// <param name="offset">Offset from the starting position</param>
/// <returns>Int</returns>
int GetInt32Reverse(unsigned char* data, int offset)
{
	int out;
	memcpy(&out, data + offset, sizeof(int));
	Reverse(&out);
	return out;
}

/// <summary>
/// Write an int into a specfic memory location
/// </summary>
/// <param name="src">Source number</param>
/// <param name="dst">Target data</param>
/// <param name="offset">Offset from start</param>
void WriteInt32(int src, unsigned char* dst, int offset)
{
	int _src = src;
	if (isLittleEndian()) Reverse(&_src);
	memcpy(dst + offset, &_src, sizeof(int));
}

/// <summary>
/// Get an int, endian tackled
/// </summary>
/// <param name="data">Data pointer</param>
/// <param name="offset">Offset</param>
/// <returns></returns>
int GetInt32(unsigned char* data, int offset)
{
	if (isLittleEndian())
		return GetInt32Reverse(data, offset);
	else
		return GetInt32NoReverse(data, offset);
}

ProcessState NetworkDataParser::ReadMNISTData(NetworkDataSet* dataSet, std::string dataPath, std::string labelPath, Network::Algorithm::NormalizationMode mode)
{
	// read file
	File dataFileObj(dataPath);
	File labelFileObj(labelPath);

	unsigned char* data, * label;

	uintmax_t dataFileSize;
	uintmax_t labelFileSize;

	dataFileObj.ReadAllBytes(&data, &dataFileSize);
	labelFileObj.ReadAllBytes(&label, &labelFileSize);

	// check file size, current limited at 4G;
	if (dataFileSize > UINT32_MAX || labelFileSize > UINT32_MAX)
	{
		return ProcessState(false, "File Too Large");
	}

	int dataMagicNumber = GetInt32(data, 0);
	int labelMagicNumber = GetInt32(label, 0);

	// check magic number
	if (dataMagicNumber != DataMagicNumber || labelMagicNumber != LabelMagicNumber)
	{		
		delete[] data;
		delete[] label;
		return ProcessState(false, "Can't Parse File");
	}

	int dataCount = GetInt32(data, DataCountOffset);
	int labelCount = GetInt32(label, LabelCountOffset);

	// check if counts match
	if (dataCount != labelCount)
	{
		delete[] data;
		delete[] label;
		return ProcessState(false, "Can't Parse File");
	}

	int dataWidth = GetInt32(data, DataWidthOffset);
	int dataHeight = GetInt32(data, DataHeightOffset);

	// push data
	for (int i = 0; i < dataCount; i++)
	{
		// NOTE: Added 2023-2-20
		NetworkData* networkdata = new NetworkData();
		networkdata->label = *(label + LabelOffset + i);
		networkdata->data = Algorithm::NormalizeData(data, DataOffset + i * dataWidth * dataHeight, dataWidth * dataHeight, mode);
		networkdata->dataSize = dataWidth * dataHeight;

		dataSet->AddData(networkdata);
	}

	dataSet->dataHeight = dataHeight;
	dataSet->dataWidth = dataWidth;

	dataSet->mode = mode;

	// free allocated spaces
	delete[] label;
	delete[] data;

	return ProcessState(true);
}

Json::Value SaveLayerJSON(NeuronLayer* layer)
{
	Json::Value root;

	// root parameters
	root["prev_count"] = layer->prevCount;
	root["bias"] = layer->bias;
	root["neuron_count"] = layer->neuronCount;

	// neuron data
	for (auto& weight : layer->weightList)
	{
		Json::Value weights;
		for (int i = 0; i < layer->prevCount; i++)
			weights.append(weight[i]);
		root["weights"].append(weights);
	}

	return root;
}

ProcessState NetworkDataParser::SaveNetworkDataJSON(FCNetwork* network, std::string path)
{
	try
	{
		ProgressTimer timer;

		Json::Value root;

		// Save Root Parameters
		root["activate_func"] = (int)network->ActivateFunc;
		root["in_count"] = network->inNeuronCount;
		root["out_count"] = network->outNeuronCount;
		root["hidden_neuron_count"] = network->hiddenNeuronCount;
		root["hidden_layer_count"] = network->hiddenLayerCount;

		// out layer
		root["out_layer_data"] = SaveLayerJSON(&network->outLayer);

		// hidden layers
		for (auto& layer : network->hiddenLayerList)
			root["hidden_layer_data"].append(SaveLayerJSON(&layer));

		// write json to string
		std::string content = Json::FastWriter().write(root);
		
		// write string to file
		if (!File(path).WriteAllText(content))
		{
			// fail
			return ProcessState(false, "Failed to save network.");
		}

		return ProcessState(true, "Network Saved.", timer.Count());
	}
	catch (std::exception e)
	{
		return ProcessState(false, std::format("Unhandled Exception: {}", e.what()));
	}
}

void ThrowLogicError(std::string what)
{
	throw Json::LogicError(what);
}

Json::Value GetMember(Json::Value& root, std::string key)
{
	if (root.isMember(key))
		return root[key];
	else
		ThrowLogicError("Can't find member: " + key);
}

Network::NeuronLayer GetLayerJSON(Json::Value val)
{
	int prevCount = GetMember(val, "prev_count").asInt();
	int neuronCount = GetMember(val, "neuron_count").asInt();
	float_n bias = GetMember(val, "bias").asFloat();

	Network::NeuronLayer layer(neuronCount, prevCount);
	
	layer.bias = bias;

	for (int i = 0; i < neuronCount; i++)
	{
		Json::Value& weights = val["weights"][i];
		for (int j = 0; j < prevCount; j++)
		{
			layer[i][j] = weights[j].asFloat();
		}
	}

	return layer;
}

ProcessState NetworkDataParser::ReadNetworkDataJSON(FCNetwork** network, std::string path)
{
	try
	{
		ProgressTimer timer;

		std::string content;

		// read string from file
		if (!File(path).ReadAllText(&content))
		{
			// fail
			return ProcessState(false, "Failed to read file!");
		}

		Json::Value root;
		
		// parse raw json string
		if (!Json::Reader().parse(content, root))
		{
			// fail
			return ProcessState(false, "Not a valid json file!");
		}

		Network::ActivateFunctionType activateFunc = (Network::ActivateFunctionType)GetMember(root, "activate_func").asInt();
		int in_count = GetMember(root, "in_count").asInt();
		int out_count = GetMember(root, "out_count").asInt();
		int hidden_neuron_count = GetMember(root, "hidden_neuron_count").asInt();
		int hidden_layer_count = GetMember(root, "hidden_layer_count").asInt();

		if (GetMember(root, "hidden_layer_data").size() != hidden_layer_count)
		{
			return ProcessState(false, "Not a valid network!");
		}

		FCNetwork* net = new FCNetwork(in_count, GetLayerJSON(GetMember(root, "out_layer_data")), hidden_neuron_count, hidden_layer_count, activateFunc);

		for (int i = 0; i < hidden_layer_count; i++)
		{
			net->hiddenLayerList.push_back(GetLayerJSON(root["hidden_layer_data"][i]));
		}

		if (*network)
		{
			(**network).Destroy();
		}

		*network = net;

		return ProcessState(true, "", timer.Count());

	}
	catch (Json::Exception e)
	{
		return ProcessState(false, std::format("Can't parse json file. Message: {}", e.what()));
	}
	catch (std::exception e)
	{
		return ProcessState(false, std::format("Unhandled Exception: {}", e.what()));
	}
}