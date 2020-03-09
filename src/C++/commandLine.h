#ifndef COMMANDLINE_H
#define COMMANDLINE_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map> 
#include <string.h>

void setHashMap(std::unordered_map<std::string, std::string> &hash, char *argv[], int argc);
void setInitialValues(std::unordered_map<std::string, std::string> &hash, std::string* iteration, std::string* outfile, std::string* inputfile, std::string *square, std::string* display);

void help();
void showVersion();


#endif