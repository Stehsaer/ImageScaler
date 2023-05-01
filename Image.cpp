#include "Image.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include<stdlib.h>
#include<format>

const float shift = 0.0;

void ShowData(float* pixels, int width, int height)
{
	unsigned char* data = (unsigned char*)malloc(sizeof(unsigned char) * width * height);

	if (data == nullptr)
		throw std::bad_alloc();

	for (int i = 0; i < width * height; i++)
	{
		data[i] = (unsigned char)(pixels[i] * 255.0);
	}

	stbi_write_png("debug.png", width, height, 1, data, 0);

	free(data);
}

YUVImage::YUVImage(std::string path)
{
	Load(path);
}

void YUVImage::Load(std::string path)
{
	int comp;
	unsigned char* data = stbi_load(path.c_str(), &width, &height, &comp, 3);

	if (data == nullptr)
	{
		throw std::exception(stbi_failure_reason());
	}

	// allocate memory
	y = new float[Count()];
	u = new float[Count()];
	v = new float[Count()];

	// do RGB->YUV conversion
	for (int index = 0; index < Count(); index++)
	{
		ColorConversion::RGB2YUV(data[index * 3] / 255.0f, data[index * 3 + 1] / 255.0f, data[index * 3 + 2] / 255.0f, y[index], u[index], v[index]);
	}

	stbi_image_free(data);
}

void YUVImage::SavePNG(std::string path)
{
	unsigned char* data = new unsigned char[Count() * 3];

	// do YUV->RGB(unsigned char) conversion
	for (int index = 0; index < Count(); index++)
	{
		ColorConversion::YUV2RGB_UC(y[index], u[index], v[index], data + index * 3);
	}

	stbi_write_png(path.c_str(), width, height, 3, data, 0);

	delete[] data;
}

void YUVImage::Free()
{
	delete[] y;
	delete[] u;
	delete[] v;
}

ImageLayer::ImageLayer(int width, int height, float* data)
{
	this->width = width;
	this->height = height;
	this->data = new float[width * height];

	if(data!=nullptr)
		memcpy(this->data, data, width * height * sizeof(float));
}

ImageLayer::ImageLayer(YUVImage& image, Channels channel)
{
	this->width = image.width;
	this->height = image.height;

	float* src = nullptr;
	switch (channel)
	{
	case Channels_Y:
		src = image.y;
		break;
	case Channels_U:
		src = image.u;
		break;
	case Channels_V:
		src = image.v;
	}

	this->data = new float[width * height];
	memcpy(data, src, width * height * sizeof(float));
}

constexpr float& ImageLayer::Get(int x, int y)
{
	return data[y * width + x];
}

ImageDataset::ImageDataset(ImageLayer& imageLayer, int x_in, int y_in, int size)
{
	hdData = new float[size * size];
	sdData = new float[size * size];

	this->size = size;

	for (int x = 0; x < size; x++)
		for (int y = 0; y < size; y++)
		{
			hdData[y * size + x] = imageLayer.Get(x_in + size / 4 + x, y_in + size / 4 + y);
		}

	for (int x = 0; x < size; x++)
		for (int y = 0; y < size; y++)
		{
			sdData[y * size + x] = imageLayer.Get(x_in + x * 2, y_in + y * 2);
			sdData[y * size + x] += imageLayer.Get(x_in + x * 2 + 1, y_in + y * 2);
			sdData[y * size + x] += imageLayer.Get(x_in + x * 2, y_in + y * 2 + 1);
			sdData[y * size + x] += imageLayer.Get(x_in + x * 2 + 1, y_in + y * 2 + 1);
			sdData[y * size + x] /= 4;
		}
}

void ImageDataset::Free()
{
	delete[] sdData;
	delete[] hdData;
}

void ImageDataset::DebugOutput()
{
	ShowData(sdData, size, size);
}