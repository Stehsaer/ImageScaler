//#include "NetworkFramework.h"
//
//#include <mutex>
//
//namespace Network::Connectivity
//{
//	void TrainWork(FullConnNetwork* network, NetworkDataSet* dataset, std::atomic<int>* currentTask, std::atomic<int>* completeTask, std::atomic<int>* maxTask, std::mutex *mutexLock)
//	{
//		std::cout << "[Train] Thread Started. Id: " << std::this_thread::get_id() << std::endl;
//
//		std::unique_ptr<float_n> target(new float_n[network->outNeuronCount]);
//
//		// create temporary instances
//		NeuronLayerInstance inLayer(&network->inLayer);
//		NeuronLayerInstance outLayer(&network->outLayer);
//
//		std::vector<NeuronLayerInstance> hiddenLayers; // hidden layer list
//		for (auto& layer : network->hiddenLayerList)
//			hiddenLayers.push_back(NeuronLayerInstance(&layer));
//
//		// main loop
//		while (1)
//		{
//			if (*maxTask < 0) // minus for quitting
//			{
//				break;
//			}
//			if (*currentTask < *maxTask) // some work waiting to quit
//			{
//				// occupy a task
//				int task = *currentTask;
//				(*currentTask)++;
//
//				if (task >= dataset->Count()) break;
//
//				// fetch bias
//				for (auto& layer : hiddenLayers)
//				{
//					layer.FetchBias();
//				}
//				outLayer.FetchBias();
//
//				const int count = hiddenLayers.size();
//
//				// Push Data (Note: Don't use PushDataFloat() in network class, use the instance class one)
//				inLayer.PushDataFloat((*dataset)[task].data);
//				// get label
//				int label = (*dataset)[task].label;
//				for (int i = 0; i < network->outNeuronCount; i++)
//				{
//					target.get()[i] = i == label ? 1.0 : 0.0;
//				}
//
//				// forward
//				for (int i = 0; i < count; i++)
//				{
//					network->ForwardTransmitLayer(hiddenLayers[i], i == 0 ? inLayer : hiddenLayers[i - 1]);
//				}
//
//				NeuronLayerInstance& lastHiddenLayer = hiddenLayers[count - 1];
//
//				for (int i = 0; i < outLayer.Count(); i++)
//				{
//					outLayer[i].value = 0.0;
//
//					for (int j = 0; j < outLayer.prevCount; j++)
//					{
//						outLayer[i].value += lastHiddenLayer[j].value * outLayer[i].weights[j];
//					}
//
//					outLayer[i].value += outLayer.bias;
//					outLayer[i].value = outLayer[i].value / (float_n)outLayer.prevCount;
//				}
//
//				// softmax layer
//				if (network->outLayerSoftMax)
//				{
//					Network::Algorithm::SoftMax(outLayer);
//					Network::Algorithm::SoftMaxGetError(outLayer, target.get());
//				}
//				else // no softmax
//				{
//					for (int i = 0; i < outLayer.Count(); i++)
//					{
//						outLayer[i].value = (*network->BackwardActive)(target.get()[i] - outLayer[i].value);
//					}
//				}
//
//				// backward transmit
//				for (int i = count - 1; i >= 0; i--)
//				{
//					network->BackwardTransmitLayer(hiddenLayers[i], i == count - 1 ? outLayer : hiddenLayers[i + 1]);
//				}
//
//				// Feedback to source
//				//mutexLock->lock(); // lock network source
//
//				for (auto& layer : hiddenLayers)
//					layer.FeedBack();
//
//				inLayer.FeedBack();
//				outLayer.FeedBack();
//
//				//mutexLock->unlock(); // unlock network source
//				(*completeTask)++;
//			}
//			else
//			{
//				std::this_thread::sleep_for(std::chrono::microseconds(100));
//			}
//		}
//
//		// clear up temporary instances
//		inLayer.Destroy();
//		outLayer.Destroy();
//		for (auto& layer : hiddenLayers)
//			layer.Destroy();
//		hiddenLayers.clear();
//
//		std::cout << "[Train] Thread Terminated. Id: " << std::this_thread::get_id() << std::endl;
//
//		return;
//	}
//
//	void TrainWorkImage(FullConnNetwork* network, std::vector<ImageDataset*>* dataset, std::atomic<int>* currentTask, std::atomic<int>* completeTask, std::atomic<int>* maxTask, std::mutex* mutexLock)
//	{
//		std::cout << "[Train] Thread Started. Id: " << std::this_thread::get_id() << std::endl;
//
//		std::unique_ptr<float_n> target(new float_n[network->outNeuronCount]);
//
//		// create temporary instances
//		NeuronLayerInstance inLayer(&network->inLayer);
//		NeuronLayerInstance outLayer(&network->outLayer);
//
//		std::vector<NeuronLayerInstance> hiddenLayers; // hidden layer list
//		for (auto& layer : network->hiddenLayerList)
//			hiddenLayers.push_back(NeuronLayerInstance(&layer));
//
//		// main loop
//		while (1)
//		{
//			if (*maxTask < 0) // minus for quitting
//			{
//				break;
//			}
//			if (*currentTask < *maxTask) // some work waiting to quit
//			{
//				// occupy a task
//				int task = *currentTask;
//				(*currentTask)++;
//
//				if (task >= dataset->size()) break;
//
//				// fetch bias
//				for (auto& layer : hiddenLayers)
//				{
//					layer.FetchBias();
//				}
//				outLayer.FetchBias();
//
//				const int count = hiddenLayers.size();
//
//				// Push Data (Note: Don't use PushDataFloat() in network class, use the instance class one)
//				inLayer.PushDataFloat((*dataset)[task]->sdData);
//				
//				for (int i = 0; i < (*dataset)[task]->size * 4; i++)
//					target.get()[i] = (*dataset)[task]->hdData[i];
//
//				// forward
//				for (int i = 0; i < count; i++)
//				{
//					network->ForwardTransmitLayer(hiddenLayers[i], i == 0 ? inLayer : hiddenLayers[i - 1]);
//				}
//
//				NeuronLayerInstance& lastHiddenLayer = hiddenLayers[count - 1];
//
//				for (int i = 0; i < outLayer.Count(); i++)
//				{
//					outLayer[i].value = 0.0;
//
//					for (int j = 0; j < outLayer.prevCount; j++)
//					{
//						outLayer[i].value += lastHiddenLayer[j].value * outLayer[i].weights[j];
//					}
//
//					outLayer[i].value += outLayer.bias;
//					outLayer[i].value = outLayer[i].value / (float_n)outLayer.prevCount;
//				}
//
//				// softmax layer
//				if (network->outLayerSoftMax)
//				{
//					Network::Algorithm::SoftMax(outLayer);
//					Network::Algorithm::SoftMaxGetError(outLayer, target.get());
//				}
//				else // no softmax
//				{
//					for (int i = 0; i < outLayer.Count(); i++)
//					{
//						outLayer[i].value = (*network->BackwardActive)(target.get()[i] - outLayer[i].value);
//					}
//				}
//
//				// backward transmit
//				for (int i = count - 1; i >= 0; i--)
//				{
//					network->BackwardTransmitLayer(hiddenLayers[i], i == count - 1 ? outLayer : hiddenLayers[i + 1]);
//				}
//
//				// Feedback to source
//				//mutexLock->lock(); // lock network source
//
//				for (auto& layer : hiddenLayers)
//					layer.FeedBack();
//
//				inLayer.FeedBack();
//				outLayer.FeedBack();
//
//				//mutexLock->unlock(); // unlock network source
//				(*completeTask)++;
//			}
//		}
//
//		// clear up temporary instances
//		inLayer.Destroy();
//		outLayer.Destroy();
//		for (auto& layer : hiddenLayers)
//			layer.Destroy();
//		hiddenLayers.clear();
//
//		std::cout << "[Train] Thread Terminated. Id: " << std::this_thread::get_id() << std::endl;
//
//		return;
//	}
//
//	void FullConnNetwork::TrainBatched(NetworkDataSet& dataset, int batchSize, float_n learningRate, std::function<void(int, int)> callback)
//	{
//		std::cout << std::format("[Train] Train Started. Parameters: batchSize={}, learningRate={}", batchSize, learningRate) << std::endl;
//
//		std::atomic<int> currentTask = 0, completedTask = 0, maxTask = 0;
//		std::mutex mainNetworkLock;
//		size_t count = dataset.Count();
//
//		std::vector<std::thread> threadList;
//
//		this->learningRate = learningRate;
//
//		// if batchSize is larger than hardware concurrency, use hardware concurrency
//		int taskCount = batchSize > std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : batchSize;
//
//		for (int i = 0; i < taskCount; i++)
//		{
//			threadList.push_back(std::thread(TrainWork, this, &dataset, &currentTask, &completedTask, &maxTask, &mainNetworkLock));
//		}
//
//		// compute
//		while (count - maxTask >= batchSize)
//		{
//
//			maxTask += batchSize;
//
//			while (completedTask < maxTask);
//
//			//mainNetworkLock.lock(); // lock main network
//
//			computeAverage(batchSize); // do average
//			UpdateWeights();
//
//			// clear values
//			inLayer.ClearValues();
//			outLayer.ClearValues();
//			for (auto& layer : hiddenLayerList)
//				layer.ClearValues();
//
//			//mainNetworkLock.unlock(); // unlock main network
//
//			callback(completedTask, count); // progress callback
//		}
//
//		// send terminate signal
//		maxTask = -1;
//
//		for (auto& thread : threadList) thread.join();
//		threadList.clear();
//	}
//
//	void FullConnNetwork::TrainBatched(std::vector<ImageDataset*>& dataset, int batchSize, float_n learningRate, std::function<void(int, int)> callback)
//	{
//		std::cout << std::format("[Train] Train Started. Parameters: batchSize={}, learningRate={}", batchSize, learningRate) << std::endl;
//
//		std::atomic<int> currentTask = 0, completedTask = 0, maxTask = 0;
//		std::mutex mainNetworkLock;
//		size_t count = dataset.size();
//
//		std::vector<std::thread> threadList;
//
//		this->learningRate = learningRate;
//
//		// if batchSize is larger than hardware concurrency, use hardware concurrency
//		int taskCount = batchSize > std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : batchSize;
//
//		for (int i = 0; i < taskCount; i++)
//		{
//			threadList.push_back(std::thread(TrainWorkImage, this, &dataset, &currentTask, &completedTask, &maxTask, &mainNetworkLock));
//		}
//
//		// compute
//		while (count - maxTask >= batchSize)
//		{
//			maxTask += batchSize;
//
//			while (completedTask < maxTask) std::this_thread::sleep_for(std::chrono::milliseconds(1));
//
//			//mainNetworkLock.lock(); // lock main network
//
//			computeAverage(batchSize); // do average
//			UpdateWeights();
//
//			// clear values
//			inLayer.ClearValues();
//			outLayer.ClearValues();
//			for (auto& layer : hiddenLayerList)
//				layer.ClearValues();
//
//			//mainNetworkLock.unlock(); // unlock main network
//
//			callback(completedTask, count); // progress callback
//			std::cerr << "Completed: " << maxTask << std::endl;
//		}
//
//		// send terminate signal
//		maxTask = -1;
//
//		for (auto& thread : threadList) thread.join();
//		threadList.clear();
//	}
//	
//}