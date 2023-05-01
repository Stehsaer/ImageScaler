#pragma once

#ifndef _NETWORK_DATA_PARSER_H_
#define _NETWORK_DATA_PARSER_H_

#include<string>
#include"NetworkFramework.h"

#include "ProcessState.h"

namespace Network
{
	class NetworkDataParser
	{
	public:
		static const int DataCountOffset = 4;
		static const int DataWidthOffset = 8;
		static const int DataHeightOffset = 12;
		static const int DataOffset = 16;

		static const int LabelCountOffset = 4;
		static const int LabelOffset = 8;

		static const int DataMagicNumber = 2051;
		static const int LabelMagicNumber = 2049;

		static ProcessState ReadMNISTData(NetworkDataSet* dataSet, std::string dataPath, std::string labelPath, Network::Algorithm::NormalizationMode mode);
		//static ProcessState SaveNetworkData(Network::Connectivity::FullConnNetwork* network, std::string path);
		//static ProcessState ReadNetworkData(Network::Connectivity::FullConnNetwork** network, std::string path);

		static ProcessState SaveNetworkDataJSON(Network::Connectivity::FullConnNetwork* network, std::string path);
		static ProcessState ReadNetworkDataJSON(Network::Connectivity::FullConnNetwork** network, std::string path);
	};
}

#endif

