#include <iostream>

#include<filesystem>
#include<random>

#include "network/Network.h"

#include "Image.h"

Network::Connectivity::FullConnNetwork* networkPtr = nullptr;
std::vector<ImageDataset*> datasets;
const int coreSize = 8;

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
	out += std::format("] {:3}%\r", (int)(progress * 100));
	std::cout << out;
}

void Train()
{
	if (!networkPtr)
	{
		std::cout << "No network loaded!" << std::endl;
		return;
	}

	float learningRate;
	int repeat;

	std::cout << "Learning Rate> ";
	std::cin >> learningRate;
	std::cout << "Repeat> ";
	std::cin >> repeat;

	std::cout << "Working... Learning rate: " << learningRate << std::endl;

	auto& network = *networkPtr;

	network.learningRate = learningRate;

	const int displayInterval = 2000;

	for (int iter = 0; iter < repeat; iter++)
	{
		std::cout << "Iteration " << iter + 1 << ", Shuffling Data..." << std::endl;

		std::shuffle(datasets.begin(), datasets.end(), std::mt19937(std::random_device()()));

		double loss = 0.0;

		for (int i = 0; i < datasets.size(); i++)
		{
			auto& data = datasets[i];

			network.PushDataFloat(data->sdData);
			network.PushTargetFloat(data->hdData);

			loss += network.GetLoss();

			network.ForwardTransmit();
			network.BackwardTransmit();
			network.UpdateWeights();

			if ((i+1) % displayInterval == 0)
			{
				std::cout << "Completed:" << (i + 1) << " Loss:" << loss / displayInterval << std::endl;
				loss = 0.0;
			}
		}

		double totalLoss = 0.0;

		std::cout << "Calculating avg loss..." << std::endl;
		for (int i = 0; i < datasets.size(); i++)
		{
			auto& data = datasets[i];

			network.PushDataFloat(data->sdData);
			network.PushTargetFloat(data->hdData);

			network.ForwardTransmit();
			totalLoss += network.GetLoss();
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

	for (int x = 0; x < newWidth; x += coreSize)
	{
		for (int y = 0; y < newHeight; y += coreSize)
		{
			// feed in data, Y
			for (int _x = 0; _x < coreSize; _x++)
				for (int _y = 0; _y < coreSize; _y++)
				{
					network.inLayer.value[_y * coreSize + _x] = yLayer.Get(x / 2 + _x, y / 2 + _y);
				}

			network.ForwardTransmit();

			for (int _x = 0; _x < coreSize; _x++)
			{
				for (int _y = 0; _y < coreSize; _y++)
				{
					output_y.Get(x + _x, y + _y) = network.outLayer.value[_y * coreSize + _x];
				}
			}

			// U
			for (int _x = 0; _x < coreSize; _x++)
				for (int _y = 0; _y < coreSize; _y++)
				{
					network.inLayer.value[_y * coreSize + _x] = uLayer.Get(x / 2 + _x, y / 2 + _y) + 0.5;
				}

			network.ForwardTransmit();

			for (int _x = 0; _x < coreSize; _x++)
			{
				for (int _y = 0; _y < coreSize; _y++)
				{
					output_u.Get(x + _x, y + _y) = network.outLayer.value[_y * coreSize + _x] - 0.5;
				}
			}

			// V
			for (int _x = 0; _x < coreSize; _x++)
				for (int _y = 0; _y < coreSize; _y++)
				{
					network.inLayer.value[_y * coreSize + _x] = vLayer.Get(x / 2 + _x, y / 2 + _y) + 0.5;
				}

			network.ForwardTransmit();

			for (int _x = 0; _x < coreSize; _x++)
			{
				for (int _y = 0; _y < coreSize; _y++)
				{
					output_v.Get(x + _x, y + _y) = network.outLayer.value[_y * coreSize + _x] - 0.5;
				}
			}
		}

		DisplayProgress((float)x / (float)newWidth);
	}

	YUVImage outputImage(newWidth, newHeight);

	outputImage.y = output_y.data;
	outputImage.u = output_u.data;
	outputImage.v = output_v.data;
	//outputImage.u = new float[newWidth * newHeight];
	//outputImage.v = new float[newWidth * newHeight];

	/*const int offset = 1;

	std::cout << std::endl << "Interpolating U,V components..." << std::endl;
	for (int x = 0; x < newWidth; x++)
	{
		for (int y = 0; y < newHeight; y++)
		{
			outputImage.u[y * newWidth + x] = srcImage.u[(y / 2 + offset) * srcImage.width + x / 2 + offset];
			outputImage.v[y * newWidth + x] = srcImage.v[(y / 2 + offset) * srcImage.width + x / 2 + offset];
		}

		DisplayProgress((float)x / (float)newWidth);
	}*/

	std::cout << std::endl << "Saving image..." << std::endl;
	outputImage.SavePNG(outputPath);

	outputImage.Free();
	srcImage.Free();

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