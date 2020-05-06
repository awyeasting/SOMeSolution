import os
import math
dirname = './Groundtruthdata/'
for file in os.listdir(dirname):
    fileInstance = open(dirname + file, "r")

    newWeightsFile = file[:-4] + '_data' + file[-4:]
    fileWriter = open(newWeightsFile, "w")
    tempLine = fileInstance.readline()
    i = 0
    while (tempLine):
        splitLine = tempLine.split(' ')
        particle_number = splitLine[0]
        file_loc = splitLine[1]
        timestamp = splitLine[2]
        particleTOF = splitLine[3]
        intensity = splitLine[4]

        year = timestamp[0:4]
        month = timestamp[4:6]
        day = timestamp[6:8]
        hour = timestamp[8:10]
        minu = timestamp[10:12]
        sec = timestamp[12:14]
        miliSec = timestamp[14:]

        dva = math.pow(311.45*(float(particleTOF) / 1000000), 2.9266)
        fileWriter.write(particleTOF + ' ')
        fileWriter.write(str(dva) + ' ')
        fileWriter.write(intensity + ' ')
        
        spectralNum = 5
        total = len(splitLine)-1
        while(spectralNum <= total):
            if spectralNum == total:
                fileWriter.write(splitLine[spectralNum])
            else:
                fileWriter.write(splitLine[spectralNum]+ ' ')
            spectralNum += 1
        i += 1
        tempLine = fileInstance.readline()
