#pragma once

#include<string>
#include<vector>
#include<random>

// for debug
void ShowData(float* pixels, int width, int height);

inline float _clamp(float x)
{
	if (x > 1.0) return 1.0;
	if (x < 0.0) return 0.0;
	return x;
}

class ColorConversion
{
public:
	inline static void YUV2RGB(float y, float u, float v, float& r, float& g, float& b)
	{
		r = y + 1.4075 * v;
		g = y - 0.3455 * u - 0.7169 * v;
		b = y + 1.779 * u;
	}

	inline static void YUV2RGB_UC(float y, float u, float v, unsigned char* data)
	{
		 
		*data = _clamp(y + 1.4075 * v) * 255;
		*(data + 1) = _clamp(y - 0.3455 * u - 0.7169 * v) * 255;
		*(data + 2) = _clamp(y + 1.779 * u) * 255;
	}

	inline static void RGB2YUV(float r, float g, float b, float& y, float& u, float& v)
	{
		y = 0.299 * r + 0.587 * g + 0.114 * b;

		u = -0.169 * r - 0.331 * g + 0.5 * b;

		v = 0.5 * r - 0.419 * g - 0.081 * b;
	}
};

enum Channels
{
	Channels_Y,
	Channels_U,
	Channels_V,
	Channels_R,
	Channels_G,
	Channels_B
};

class YUVImage
{
public:
	float* y, * u, * v;
	int width, height;

	YUVImage(int width, int height) : width(width), height(height), y(nullptr), u(nullptr), v(nullptr) {};
	YUVImage(std::string path);
	
	void Load(std::string path);
	void SavePNG(std::string path);

	void FreeData();

	inline int GetOffset(int x, int y)
	{
		return y * width + x;
	}

	inline constexpr int Count()
	{
		return width * height;
	}
};

class ImageLayer
{
public:
	float* data;
	int width, height;

	ImageLayer(int width, int height, float* data = nullptr);
	ImageLayer(YUVImage& image, Channels channel);

	constexpr float& Get(int x, int y);
	void FreeData();
};

struct ImageDataset
{
public:
	float* hdData;
	float* sdData;

	int size; // sd resolution, square shape
	const int scaleCoeff = 2;

	ImageDataset(ImageLayer& imageLayer, int x, int y, int size);
	void Free();

	void DebugOutput();
};

inline void GenDataset(std::vector<ImageDataset*>& datasets, std::string path, int count, int coreSize, Channels channel)
{
	YUVImage image(path);
	ImageLayer layer(image, channel);

	std::random_device device;
	std::mt19937 gen(device());

	std::uniform_int_distribution<int> distX(coreSize*2, image.width - coreSize * 2);
	std::uniform_int_distribution<int> distY(coreSize*2, image.height - coreSize * 2);

	datasets.reserve(datasets.size() + count);

	for (int i = 0; i < count; i++)
	{
		datasets.push_back(new ImageDataset(layer, distX(gen), distY(gen), coreSize));
	}
	delete[] layer.data;
	image.FreeData();
}

extern const float shift;