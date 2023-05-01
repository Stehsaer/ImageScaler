#ifndef _NETWORK_ALGORITHM_H_
#define _NETWORK_ALGORITHM_H_

#include "NetworkStructure.h"

namespace Network
{
	typedef float_n(*ActivateFunction)(float_n);

	namespace Algorithm
	{
		enum NormalizationMode
		{
			ZeroToOne = 0,
			MinusOneToOne = 1,
			NoNormalization = -1
		};

		/// <summary>
		/// Sigmoid
		/// </summary>
		float_n Sigmoid(float_n x);

		/// <summary>
		/// Derivative of Sigmoid
		/// </summary>
		float_n Sigmoid_D(float_n x);

		/// <summary>
		/// Shifted Sigmoid
		/// </summary>
		float_n ShiftedSigmoid(float_n x);

		/// <summary>
		/// D: Shifted Sigmoid
		/// </summary>
		float_n ShiftedSigmoid_D(float_n x);

		/// <summary>
		/// ReLU
		/// </summary>
		float_n ReLU(float_n x);

		/// <summary>
		/// D: ReLU
		/// </summary>
		float_n ReLU_D(float_n x);

		/// <summary>
		/// Leaky ReLU
		/// </summary>
		float_n LeakyReLU(float_n x);

		/// <summary>
		/// D: Leaky ReLU
		/// </summary>
		float_n LeakyReLU_D(float_n x);

		float_n Linear(float_n x);
		float_n Linear_D(float_n x);

		/// <summary>
		/// Normalize data in a given data range
		/// </summary>
		/// <param name="data">Original Data</param>
		/// <param name="offset">Offset</param>
		/// <param name="dataSize">Size of the data</param>
		/// <param name="mode">Normalization Mode</param>
		/// <returns>Processed Data Pointer</returns>
		float* NormalizeData(unsigned char* data, int offset, int dataSize, NormalizationMode mode);

		/// <summary>
		/// Softmax
		/// </summary>
		void SoftMax(NeuronLayer& layer);
		void SoftMax(NeuronLayerInstance& layer);

		/// <summary>
		/// Softmax
		/// </summary>
		/// <param name="target">Target Data used for error calculation</param>
		void SoftMaxGetError(NeuronLayer& layer, float_n* target);
		void SoftMaxGetError(NeuronLayerInstance& layer, float_n* target);
	}

	enum class ActivateFunctionType :int
	{
		Sigmoid, SigmoidShifted, ReLU, LeakyReLU, Linear
	};

	const int ActivateFunctionCount = 4;

	inline const Network::ActivateFunction forwardFuncList[] =
	{ &Network::Algorithm::Sigmoid,
	&Network::Algorithm::ShiftedSigmoid ,
	&Network::Algorithm::ReLU ,
	&Network::Algorithm::LeakyReLU,
		&Network::Algorithm::Linear
	};

	inline const Network::ActivateFunction backwardFuncList[] =
	{ &Network::Algorithm::Sigmoid_D,
	&Network::Algorithm::ShiftedSigmoid_D ,
	&Network::Algorithm::ReLU_D ,
	&Network::Algorithm::LeakyReLU_D,
		&Network::Algorithm::Linear_D
	};

	inline const char* functionNameList[] =
	{
		"Sigmoid",
		"Tanh",
		"ReLU",
		"LeakyReLU",
		"Linear"
	};
}

#endif
