#include "NetworkData.h"
//#include "json/json.h"

#include "NetworkAlgorithm.h"

#include <algorithm>
#include <random>

using namespace std::filesystem;

using namespace Network;

NetworkData::NetworkData()
{
	this->data = nullptr;
	this->label = -1;
	this->dataSize = -1;
}

void NetworkData::DrawContentDebug(int width, Algorithm::NormalizationMode mode)
{
	const std::string chars[] = { "  ", "██" };

	printf("\nContent of data: (label %d)\n", label);

	switch (mode)
	{
	case Algorithm::ZeroToOne:
		for (int i = 0; i < dataSize; i++)
		{
			printf(data[i] > 0.2? chars[1].c_str() : chars[0].c_str()); // print string into console
			
			if (i % width == width - 1)
			{
				printf("\n");
			}
		}
		break;

	case Algorithm::MinusOneToOne:
		for (int i = 0; i < dataSize; i++)
		{
			printf(data[i] > -0.6 ? chars[1].c_str() : chars[0].c_str()); // print string into console
			
			if (i % width == width - 1)
			{
				printf("\n");
			}
		}
		break;

	case Algorithm::NoNormalization:
		for (int i = 0; i < dataSize; i++)
		{
			printf(data[i] > 20 ? chars[1].c_str() : chars[0].c_str()); // print string into console
			
			if (i % width == width - 1)
			{
				printf("\n");
			}
		}
		break;
	}
}

int NetworkDataSet::Count()
{
	return dataSet.size();
}

void NetworkDataSet::AddData(NetworkData* data)
{
	dataSet.push_back(data);
}

void NetworkDataSet::Destroy()
{
	for (auto data : dataSet)
	{
		free(data->data);
		delete data;
	}

	dataSet.clear();
}

NetworkData& NetworkDataSet::operator[](int index)
{
	return *dataSet[index];
}

void NetworkDataSet::Shuffle()
{
	auto rng = std::default_random_engine{}; // random generator
	std::shuffle(dataSet.begin(), dataSet.end(), rng);
}

// Flip XY for every image. Used for converting column-wise to row-wise
void NetworkDataSet::FlipXY()
{
	for (auto data : dataSet)
	{
		float* result = new float[dataWidth * dataHeight]; // space to store flip result

		for(int x = 0; x < dataWidth; x++)
			for (int y = 0; y < dataHeight; y++)
			{
				result[y * dataWidth + x] = data->data[x * dataHeight + y];
			}

		delete[] data->data; // delete the data before flip
		data->data = result;
	}
}