#include "commandLine.h"

// List of commands
std::string cmd[] = {"--help","-h", "-N", "--trainingiterations","-i", "--input", "-v", "--version",""};


void showVersion()
{
    std::cout << "version 1.0.0" << std::endl;
}


void help()
{
    std::string line;
    std::ifstream helpfile;
    helpfile.open ("help_info.txt");

    if(helpfile.is_open())
    {
        while(!helpfile.eof())
        {
            std::getline(helpfile, line);
            std::cout << line << std::endl;
        }
    }
    else
    {
        std::cout << "Could not open file." << std::endl;
    }
}

void setHashMap(std::unordered_map<std::string, std::string> &hash, char *argv[], int argc)
{

    for(int i  = 0; i < argc; i++)
	{   
		if(strcmp(argv[i], "-N") == 0 || strcmp(argv[i], "--trainingiterations") == 0)
		{
			hash[argv[i]] = argv[i + 1];
		}
		else if(strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--out") == 0)
		{
			hash[argv[i]] = argv[i + 1];
		}
		else if(strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0)
		{
			hash[argv[i]] = argv[i + 1];
		}
		else if(strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--squareneurons") == 0)
		{
			hash[argv[i]] = argv[i + 1];
		}
		else if(strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--display") == 0)
		{
			hash[argv[i]] = argv[i + 1];
		}
	}
}

void setInitialValues(std::unordered_map<std::string, std::string> &hash, std::string* iteration, std::string* outfile, std::string* inputfile, std::string *square, std::string* display)
{
    if(hash["-N"].size() > 0)
    {
        *iteration = hash["-N"];
    }
    else if(hash["--iterationcount"].size() > 0)
    {
        *iteration = hash["--iterationcount"];
    }

    if(hash["-o"].size() > 0)
    {
        *outfile = hash["-o"];
    }
    else if(hash["--out"].size() > 0)
    {
        *outfile = hash["--out"];
    }

    if(hash["-i"].size() > 0)
    {
        *inputfile = hash["-i"];
    }
    else if(hash["--input"].size() > 0)
    {
        *inputfile = hash["--input"];
    }

    if(hash["-s"].size() > 0)
    {
        *square = hash["-s"];
    }
    else if(hash["--squareneurons"].size() > 0)
    {
        *square = hash["--squareneurons"];
    }

    if(hash["-d"].size() > 0)
    {
        *display = hash["-d"];
    }
    else if(hash["--display"].size() > 0)
    {
        *display = hash["--display"];
    }
}

int findCommand(std::string argv)
{
    int index = 0;

    while(cmd[index] != ""){
        if (cmd[index] == argv)
        {
            return index;
        }
        index++;
    }
    return -1;
}