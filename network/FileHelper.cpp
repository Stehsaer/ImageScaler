#include "FileHelper.h"
#include <iostream>
#include <fstream>

using namespace std::filesystem;

File::File(std::string _path)
{
	filePath = path(_path);
}

bool File::Exists()
{
	return exists(filePath) && is_regular_file(filePath);
}

size_t File::Size()
{
	return file_size(filePath);
}

bool File::ReadAllBytes(unsigned char** dst, size_t* size)
{
	std::ifstream stream = std::ifstream(filePath.string(), std::ios::binary);

	if (stream.is_open())
	{
		*size = Size();

		*dst = new unsigned char[*size];

		if (!*dst)
		{
			return false;
		}

		stream.read((char*)*dst, *size);
		stream.close();

		return true;
	}
	else
		return false;
}

bool File::WriteAllBytes(unsigned char* src, size_t size)
{
	// remove existing file
	if (Exists()) std::filesystem::remove(filePath);

	std::ofstream stream = std::ofstream(filePath.string(), std::ios::binary);

	if (stream.is_open())
	{
		stream.seekp(std::ios::beg);
		stream.write((const char*)src, size);
		stream.close();

		return true;
	}
	else
		return false;
}

bool File::WriteAllText(std::string txt)
{
	// remove existing file
	if (Exists()) std::filesystem::remove(filePath);

	std::ofstream stream = std::ofstream(filePath.string());

	if (stream.is_open())
	{
		stream << txt;
		stream.close();

		return true;
	}
	else
		return false;
}

bool File::ReadAllText(std::string* dst)
{
	std::ifstream stream = std::ifstream(filePath.string());

	if (stream.is_open())
	{
		std::string str;
		while (std::getline(stream, str))
			*dst += str + "\n";

		stream.close();

		return true;
	}
	else
		return false;
}

FileInfo::FileInfo(std::string path)
{
	std::filesystem::path pth(path);
	fullPath = path;
	size = file_size(pth);
	name = pth.filename().string();
}