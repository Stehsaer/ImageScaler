#ifndef _PROCESS_STATE_H_
#define _PROCESS_STATE_H_
#include <string>

struct ProcessState // A struct to indicate success or not
{
	std::string msg;
	bool success;
	long duration;

	ProcessState(bool success, std::string msg = "", long duration = -1) : msg(msg), success(success), duration(duration) {}
};

#endif