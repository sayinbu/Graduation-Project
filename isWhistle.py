import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from scipy.interpolate import interp1d

##############################

minFrequency = 600
maxFrequency = 5000

minIntensity = 100.0
maxIntensity = 100000.0

minStandardDeviation = 0.1
maxStandardDeviation = 1.0

highPass = 100
lowPass = 10000

minNumZeroCross = 50
maxNumZeroCross = 200

numRobust = 10

##############################

def findNearestNonPeakPoints(array, index):
    medianCandidates = np.array([])
    rightIndex = index
    leftIndex = index
    
    while len(medianCandidates) < 4:
        rightIndex += 1
        if rightIndex < len(array) and array[rightIndex] < 5000:
            medianCandidates = np.append(medianCandidates, [array[rightIndex]])
        leftIndex -= 1
        if leftIndex >= 0 and array[leftIndex] < 5000:
            medianCandidates = np.append(medianCandidates, [array[leftIndex]])
    return medianCandidates

def clearNonWhistlePoints(array):
    for i in range(len(array)):
        if array[i] > 5000:
            medianCandidates = findNearestNonPeakPoints(array, i)
            array[i] =np.median(medianCandidates)

def deleteInitialZeros(array,t):
    numOfZeros = 0
    for i in range(len(array)):
        if array[i] == 0:
            numOfZeros += 1
        else:
            break
    return (array[numOfZeros:],t[numOfZeros:])

def deleteLastZeros(array,t):
    numOfZeros = 0
    for i in reversed(range(len(array))):
        if array[i] == 0:
            numOfZeros += 1
        else:
            break
    if numOfZeros == 0:
        return (array,t)
    else:
        return (array[:-numOfZeros],t[:-numOfZeros])

################################################################

sampFreq, snd = wavfile.read('whistles/anten2.wav')

#snd.astype('float')

numOfPoints, numOfChannels = snd.shape

monoChannelSound = (snd[:,0] + snd[:,1]) / 2

#totalTime = numOfPoints / sampFreq

timeArray = np.arange(0, numOfPoints, 1)
timeArray = (timeArray / sampFreq)


f, t, Sxx = signal.spectrogram(monoChannelSound, sampFreq)
for i in range(0, len(f)):
    if f[i] > 5000:
        x = (i)
        newF = f[:x]
        print(f)
        print(f[:x])
        newSxx = Sxx[:x]
        print(newF.shape)
        print(newSxx.shape)
        break

plt.pcolormesh(t, newF, newSxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

highFrequencies = np.array([])
for column in Sxx.T:
    frequencyOfHighestIntensity = f[np.argmax(column)]
    highFrequencies = np.append(highFrequencies, [frequencyOfHighestIntensity])

#############################################
# frequency check
#
# print("Total: " + str(len(highFrequencies)))
#
# counter = 0
# for i in highFrequencies:
#     if i > minFrequency and i < maxFrequency:
#         counter += 1
#
# print("Total Passed: " + str(counter))
##############################################

clearNonWhistlePoints(highFrequencies)
highFrequencies, t = deleteInitialZeros(highFrequencies,t)
highFrequencies, t = deleteLastZeros(highFrequencies,t)
f = interp1d(t, highFrequencies)

plt.plot(t, f(t))
plt.show()

# plt.pcolormesh(t, f, Sxx)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()

# mpl.rcParams['agg.path.chunksize'] = 100000
# plt.plot(timeArray, monoChannelSound)
# plt.show()