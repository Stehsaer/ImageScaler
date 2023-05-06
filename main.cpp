#include <iostream>
#include<filesystem>
#include<random>
#include <omp.h>
#include <assert.h>

#include "network/Network.h"
#include "Image.h"
#include "network/ProgressTimer.h"

Network::Connectivity::FullConnNetwork* networkPtr = nullptr;
std::vector<ImageDataset*> datasets;
const int coreSize = 8;

void PrintValues(float* value, int count)
{
	for (int i = 0; i < count; i++)
		printf("[%f]", value[i]);
	printf("\n");
}

void DisplayProgress(float progress)
{
	std::string out = "[";
	int x = 40 * progress;
	for (int i = 0; i < 40; i++)
	{
		if (i < x)
			out += "=";
		else
			out += " ";
	}
	out += std::format("] {:5.1f}%\r", progress * 100);
	std::cerr << out;
}

void Train()
{
	if (!networkPtr)
	{
		std::cout << "No network loaded!" << std::endl;
		return;
	}

	float learningRate;
	int repeat, batchSize;

	std::cout << "Learning Rate> ";
	std::cin >> learningRate;
	std::cout << "Repeat> ";
	std::cin >> repeat;
	std::cout << "Batch Size> ";
	std::cin >> batchSize;

	std::cout << "Working..." << std::endl;

	auto& network = *networkPtr;

	network.learningRate = learningRate;

	for (int iter = 0; iter < repeat; iter++)
	{
		std::cout << "Iteration " << iter + 1 << ", Shuffling Data..." << std::endl;

		std::shuffle(datasets.begin(), datasets.end(), std::mt19937(std::random_device()()));

		if (batchSize == 1)
		{
			for (int i = 0; i < datasets.size(); i++)
			{
				auto& data = datasets[i];

				network.PushDataFloat(data->sdData);
				network.PushTargetFloat(data->hdData);
				network.ForwardTransmit();
				network.BackwardTransmit();
				network.UpdateWeights();

				if(i % 500 == 0)
					DisplayProgress((float)i / datasets.size());
			}
		}
		else
		{
			Network::Connectivity::FullConnNetworkInstance instance(networkPtr);

			for (int i = 0; i < datasets.size(); i++)
			{
				auto& data = datasets[i];

				instance.FetchBias();
				instance.PushData(data->sdData);
				instance.PushTarget(data->hdData);

				instance.ForwardTransmit();
				instance.BackwardTransmit();

				instance.FeedBack();

				if (i % batchSize == 0)
				{
					network.computeAverage(batchSize);
					network.UpdateWeights();
					network.ClearSum();

					if (i / batchSize % 15 == 0)
						DisplayProgress((float)i / datasets.size());
				}
			}

			instance.FreeData();
		}

		std::atomic<double> totalLoss = 0.0;

		std::cout << std::endl << "Calculating avg loss..." << std::endl;

		std::vector<Network::Connectivity::FullConnNetworkInstance*> instances;
		for (int i = 0; i < omp_get_max_threads(); i++)
			instances.push_back(new Network::Connectivity::FullConnNetworkInstance(networkPtr));

#pragma omp parallel for
		for (int i = 0; i < datasets.size(); i++)
		{
			/*auto& data = datasets[i];

			network.PushDataFloat(data->sdData);
			network.PushTargetFloat(data->hdData);

			network.ForwardTransmit();
			totalLoss += network.GetLoss();*/

			auto& instance = *instances[omp_get_thread_num()];
			instance.FetchBias();

			auto& data = datasets[i];

			instance.PushData(data->sdData);
			instance.PushTarget(data->hdData);
			instance.ForwardTransmit();

			totalLoss += instance.GetLoss();
		}

		for (auto& item : instances)
		{
			item->FreeData();
			delete item;
		}

		std::cout << "Avg Loss: " << totalLoss / datasets.size() << std::endl;
	}

	std::cout << "Done." << std::endl;
}

void NewNetwork()
{
	int hiddenLayerCount, hiddenNeuronCount;
	std::cout << "Hidden-Layer-Count> ";
	std::cin >> hiddenLayerCount;
	std::cout << "Hidden-Neuron-Count> ";
	std::cin >> hiddenNeuronCount;

	std::cout << "Working..." << std::endl;

	if (networkPtr)
	{
		networkPtr->Destroy();
		delete networkPtr;
	}

	networkPtr = new Network::Connectivity::FullConnNetwork(coreSize * coreSize, coreSize * coreSize, hiddenNeuronCount, hiddenLayerCount, Network::ActivateFunctionType::LeakyReLU, 0.0, false);
	networkPtr->RandomizeAllWeights(-0.9, 0.9);

	std::cout << "Done." << std::endl;
}

void Scale()
{
	if (!networkPtr)
	{
		std::cout << "No network loaded!" << std::endl;
		return;
	}

	std::string path;
	std::string outputPath;

	std::cout << "Source> ";
	std::cin >> path;
	std::cout << "Output> ";
	std::cin >> outputPath;

	std::cout << "Working..." << std::endl;

	std::cout << "Reading source image..." << std::endl;
	auto& network = *networkPtr;

	ProgressTimer timer;

	YUVImage srcImage(path);
	ImageLayer yLayer(srcImage, Channels_Y);
	ImageLayer uLayer(srcImage, Channels_U);
	ImageLayer vLayer(srcImage, Channels_V);

	int newWidth = (srcImage.width - coreSize / 2) / coreSize * (coreSize * 2);
	int newHeight = (srcImage.height - coreSize / 2) / coreSize * (coreSize * 2);
	std::cout << "Size:" << newWidth << "," << newHeight << std::endl;

	ImageLayer output_y(newWidth, newHeight);
	ImageLayer output_u(newWidth, newHeight);
	ImageLayer output_v(newWidth, newHeight);

	std::atomic<int> progress = 0;

	std::vector<Network::Connectivity::FullConnNetworkInstance*> tempNetworks;

	int threadCount = omp_get_max_threads();
	for (int i = 0; i < threadCount; i++)
	{
		tempNetworks.push_back(new Network::Connectivity::FullConnNetworkInstance(networkPtr));
	}

#pragma omp parallel for
	for (int x = 0; x < newWidth; x += coreSize)
	{
		using namespace Network;

		int threadnum = omp_get_thread_num();

		auto& nwk = *tempNetworks[threadnum];

		for (int y = 0; y < output_y.height; y += coreSize)
		{
			// feed in data, Y
			for (int _x = 0; _x < coreSize; _x++)
				for (int _y = 0; _y < coreSize; _y++)
				{
					nwk.inLayer.value[_y * coreSize + _x] = yLayer.Get(x / 2 + _x, y / 2 + _y);
				}

			network.ForwardTransmit(nwk.inLayer, nwk.outLayer, nwk.hiddenLayerList);

			for (int _x = 0; _x < coreSize; _x++)
			{
				for (int _y = 0; _y < coreSize; _y++)
				{
					output_y.Get(x + _x, y + _y) = nwk.outLayer.value[_y * coreSize + _x];
				}
			}

			// U
			for (int _x = 0; _x < coreSize; _x++)
				for (int _y = 0; _y < coreSize; _y++)
				{
					nwk.inLayer.value[_y * coreSize + _x] = uLayer.Get(x / 2 + _x, y / 2 + _y) + 0.5;
				}

			network.ForwardTransmit(nwk.inLayer, nwk.outLayer, nwk.hiddenLayerList);

			for (int _x = 0; _x < coreSize; _x++)
			{
				for (int _y = 0; _y < coreSize; _y++)
				{
					output_u.Get(x + _x, y + _y) = nwk.outLayer.value[_y * coreSize + _x] - 0.5;
				}
			}

			// V
			for (int _x = 0; _x < coreSize; _x++)
				for (int _y = 0; _y < coreSize; _y++)
				{
					nwk.inLayer.value[_y * coreSize + _x] = vLayer.Get(x / 2 + _x, y / 2 + _y) + 0.5;
				}

			network.ForwardTransmit(nwk.inLayer, nwk.outLayer, nwk.hiddenLayerList);

			for (int _x = 0; _x < coreSize; _x++)
			{
				for (int _y = 0; _y < coreSize; _y++)
				{
					output_v.Get(x + _x, y + _y) = nwk.outLayer.value[_y * coreSize + _x] - 0.5;
				}
			}
		}

		progress += coreSize;

		if ((progress / coreSize) % 20 == 0)
			DisplayProgress((float)progress / (float)newWidth);

		//printf("x=%d, time=%dus\n", x, timer.CountAndReset() / 1000LL);
	}

	// clean up
	for (auto& item : tempNetworks)
	{
		item->FreeData();
		delete item;
	}
	yLayer.FreeData();
	uLayer.FreeData();
	vLayer.FreeData();
	srcImage.FreeData();

	auto ms = timer.CountMs();
	std::cout << std::endl << std::format("Scaling Time: {}ms", ms) << std::endl;

	// save image
	YUVImage outputImage(newWidth, newHeight);

	outputImage.y = output_y.data;
	outputImage.u = output_u.data;
	outputImage.v = output_v.data;

	std::cout << "Saving image..." << std::endl;
	outputImage.SavePNG(outputPath);

	outputImage.FreeData();

	std::cout << "Done." << std::endl;
}

void Load()
{
	std::string path;
	std::cout << "Path> ";
	std::cin >> path;

	std::cout << "Working..." << std::endl;

	ProcessState state = Network::NetworkDataParser::ReadNetworkDataJSON(&networkPtr, path);
	if (!state.success)
	{
		std::cout << "Failed. Message: " << state.msg;
		return;
	}

	networkPtr->outLayerSoftMax = false;

	std::cout << "Done." << std::endl;
}

void Save()
{
	if (!networkPtr)
	{
		std::cout << "No network loaded!" << std::endl;
		return;
	}

	std::string path;
	std::cout << "Path> ";
	std::cin >> path;

	std::cout << "Working..." << std::endl;
	Network::NetworkDataParser::SaveNetworkDataJSON(networkPtr, path);
	std::cout << "Done." << std::endl;
}

void AddDataset()
{
	std::string path;
	int count;

	std::cout << "Path> ";
	std::cin >> path;
	std::cout << "Count> ";
	std::cin >> count;

	std::cout << "Working..." << std::endl;
	GenDataset(datasets, path, count, coreSize, Channels_Y);
	std::cout << "Done." << std::endl;
}

void ClearDataset()
{
	std::string yes;
	std::cout << "Confirm? Enter \"Yes\" to continue> ";
	std::cin >> yes;

	if (yes == "Yes")
	{
		std::cout << "Working..." << std::endl;
		for (auto& data : datasets)
		{
			data->Free();
		}
		datasets.clear();
		datasets.shrink_to_fit();

		std::cout << "Done." << std::endl;
	}
	else
	{
		std::cout << "Operation Cancelled." << std::endl;
	}
}

int main()
{
	std::cout << "Image Scaler by Stehsaer" << std::endl;
	try
	{
		while (1)
		{
			std::string command;
			std::cout << "> ";
			std::cin >> command;

			if (command == "train")
			{
				Train();
			}
			else if (command == "new")
			{
				NewNetwork();
			}
			else if (command == "scale")
			{
				Scale();
			}
			else if (command == "load")
			{
				Load();
			}
			else if (command == "save")
			{
				Save();
			}
			else if (command == "add_dataset")
			{
				AddDataset();
			}
			else if (command == "clear_dataset")
			{
				ClearDataset();
			}
			else if (command == "exit")
			{
				return EXIT_SUCCESS;
			}
			std::cout << std::endl;
		}
	}
	catch (std::exception e)
	{
		std::cout << "Uncaught Exception: " << e.what() << std::endl;
	}

	return EXIT_SUCCESS;
}