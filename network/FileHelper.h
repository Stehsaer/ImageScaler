#ifndef _FILE_HELPER_H_
#define _FILE_HELPER_H_

#include <string>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <iomanip>

class File
{
public:
	std::filesystem::path filePath;

	File(std::string path);

	bool Exists();
	size_t Size();

	bool ReadAllText(std::string* dst);
	bool ReadAllBytes(unsigned char** dst, size_t* size);
	
	bool WriteAllText(std::string txt);
	bool WriteAllBytes(unsigned char* src, size_t size);
};

struct FileInfo
{
public:
	std::string name;
	std::string fullPath;
	size_t size;

	FileInfo(std::string path);
};

enum class SizeFormat
{
	Byte,
	KiloByte,
	MegaByte,
	GigaByte,
	TeraByte
};

inline std::string FormatFileSize(size_t size, int decimal, SizeFormat format)
{
	const std::string unit[] = { "B", "KB", "MB", "GB", "TB" };
	double convertedSize = size;
	convertedSize /= 1024 ^ (int)format;

	std::stringstream stream;
	stream.precision(decimal);
	stream.setf(std::ios::fixed);

	stream << convertedSize << unit[(int)format];

	return stream.str();
}

#endif