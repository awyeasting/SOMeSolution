# SOMeSolution
An iteratively developed approach to the problem of fast SOM training. Will work towards the implementation of the HPSOM algorithm described by Liu et al. 

## C++ install

To compile the code to a library,
```bash
cd ~/SOMeSolution/src/C++
make
```
The static library will be in bin/somesolution.a

To compile the code to a commandline usable executable,
```bash
cd ~/SOMeSolution/src/C++
make build
```

## Commandline Usage

Through the command line you can add different flags and optional arguments.

Arguments:
```bash
Positional Arguments                    Description
WIDTH HEIGHT                            Sets the width and heigth of the SOM

Flag                                    Description
-v, --version                           View the current version of SOMeSolution
-o, --out     FILENAME                  Specify what the output file location should be
-e, --epoch   NUMBER                    Specifies the number of ephocs used in training. Default = 10
-g, -generate NUMBER NUMBER             Sets the number of examples and dimensions      
-i, --input   FILENAME                  Specifies the file that should be trained
```

Example:
The following will make a 10 x 10 SOM, generate it's own training data, have 100 features and 100 dimensions
```bash
somesolution 10 10 -g 100 100 -o trained_data.txt
```

## Python Visualization

To visualize a SOM weights file produced by the commandline executable, simply run:
```bash
python som.py -i weights.txt -d <display method>
```

## License 
[GPLv3](https://choosealicense.com/licenses/lgpl-3.0/)
