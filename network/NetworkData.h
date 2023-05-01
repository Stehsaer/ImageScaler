#pragma once

#ifndef _NETWORK_DATA_H_
#define _NETWORK_DATA_H

#include <filesystem>
#include<vector>
#include<string>
#include<iostream>

#include "NetworkStructure.h"
#include "NetworkAlgorithm.h"

namespace Network
{
	struct NetworkData
	{
	public:
		float* data;
		int label, dataSize;

		NetworkData();
		void DrawContentDebug(int width, Algorithm::NormalizationMode mode);
	};

	class NetworkDataSet
	{
	public:
		std::vector<NetworkData*> dataSet; // Stores all pointers pointing to the NetworkData in the set
		int dataWidth, dataHeight; // Width and height of the data image
		std::string name; // Name for the set

		Network::Algorithm::NormalizationMode mode;

		NetworkDataSet(std::string name = "") : dataWidth(0), dataHeight(0), name(name) {}
		int Count();
		void AddData(NetworkData* data);
		NetworkData& operator[](int index);
		void Destroy(); // clear all data in dataSet
		void Shuffle(); // shuffle all the data
		void FlipXY();
	};
}

#endif
