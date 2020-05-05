# SOMeSolution
An iteratively developed approach to the problem of fast SOM training. Will work towards the implementation of the HPSOM algorithm described by Liu et al. 

## Installation

Install SOMeSolution by downloading the git repo. Unzip the folders.

## Compile

To compile the code, cd to the folder C++. The file path:
```bash
cd ~/SOMeSolution/src/C++
```
Use the follow command to compile:
```bash
g++ SOM_work.cpp SOM.cpp -o main -fopenmp
```

## Usage

Through the command line you can add different flags and optional arguments.

Arguments:
```bash
Positional Arguments                    Description
WIDTH HEIGHT                            Sets the width and heigth of the SOM

Flag                                    Description
-v, --version                           View the current version of SOMeSolution
-o, --out     FILENAME                  Specify what the output file location should be
-e, --epoch   NUMBER                    Specifies the number of ephocs used in training. Default = 1000
-g, -generate NUMBER NUMBER             Sets the number of examples and dimensions      
-i, --input   FILENAME                  Specifies the file that should be trained
```

Example:
```bash
main 10 10 -g 100 100 -o trained_data.txt
```
The following will make a 10 x 10 SOM map, generate it's own training data, have 100 features and 100 dimensions

## Licsense 
[GPLv3](https://choosealicense.com/licenses/GPLv3/)
