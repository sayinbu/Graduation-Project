# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from scipy.io import wavfile
# from scipy import signal
#
# sampFreq, snd = wavfile.read('whistles/alimert1.wav')
#
# #snd.astype('float')
#
# numOfPoints, numOfChannels = snd.shape
#
# monoChannelSound = (snd[:,0] + snd[:,1]) / 2
#
# #totalTime = numOfPoints / sampFreq
#
# timeArray = np.arange(0, numOfPoints, 1)
# timeArray = (timeArray / sampFreq) * 1000
#
# f, t, Sxx = signal.spectrogram(monoChannelSound, sampFreq, nperseg=512)
# #f, t, Sxx = signal.spectrogram(monoChannelSound, sampFreq, nperseg=512) resolution
#
# plt.pcolormesh(t, f, Sxx)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()
#
# # mpl.rcParams['agg.path.chunksize'] = 100000
# # plt.plot(timeArray, monoChannelSound)
# # plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from scipy.interpolate import interp1d
import itertools
import operator

import numpy


def smooth(x, window_len=11, window='hanning'):
    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = numpy.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = numpy.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')

    y = numpy.convolve(w / w.sum(), s, mode='valid')
    return y


class Whistle:

    def __init__(self, filename):
        spectrogram = self.__getSoundSpectrogram(filename)
        t, leadingFreqsOverTime = self.__extractLeadingFrequenciesAndTime(spectrogram)
        self.__clearNonWhistlePoints(leadingFreqsOverTime)
        leadingFreqsOverTime, t = self.__deleteInitialZeros(leadingFreqsOverTime, t)
        leadingFreqsOverTime, t = self.__deleteLastZeros(leadingFreqsOverTime, t)
        self.t = np.floor(t * 100000) / 100000
        self.leadingFreqsOverTime = leadingFreqsOverTime
        self.t = np.concatenate([[0], self.t])
        self.leadingFreqsOverTime = np.concatenate([[0], self.leadingFreqsOverTime])

    def __getSoundSpectrogram(self, filename):
        sampFreq, snd = wavfile.read(filename)
        numOfPoints, numOfChannels = snd.shape
        if numOfChannels == 2:
            monoChannelSound = (snd[:, 0] + snd[:, 1]) / 2
        else:
            monoChannelSound = snd[:, 0]
        return signal.spectrogram(monoChannelSound, sampFreq)

    def __extractLeadingFrequenciesAndTime(self, spectrogram):
        f, t, Sxx = spectrogram
        highFrequencies = np.array([])
        for column in Sxx.T:
            frequencyOfHighestIntensity = f[np.argmax(column)]
            highFrequencies = np.append(highFrequencies, [frequencyOfHighestIntensity])
        return (t,highFrequencies)

    def __findNearestNonPeakPoints(self, array, index):
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

    def __clearNonWhistlePoints(self, array):
        for i in range(len(array)):
            if array[i] > 5000:
                medianCandidates = self.__findNearestNonPeakPoints(array, i)
                array[i] = np.median(medianCandidates)
            if array[i] < 600:
                array[i] = 600
        return array

    def __deleteInitialZeros(self, array, t):
        numOfZeros = 0
        for i in range(len(array)):
            if array[i] == 0:
                numOfZeros += 1
            else:
                break
        return (array[numOfZeros:], t[numOfZeros:])

    def __deleteLastZeros(self, array, t):
        numOfZeros = 0
        for i in reversed(range(len(array))):
            if array[i] == 0:
                numOfZeros += 1
            else:
                break
        if numOfZeros == 0:
            return (array, t)
        else:
            return (array[:-numOfZeros], t[:-numOfZeros])

    def getCorrWith(self, whistle):
        f = interp1d(whistle.t, whistle.leadingFreqsOverTime)
        squeezedTime = np.floor(((self.t * whistle.t[-1]) / self.t[-1])* 100000) / 100000
        return np.corrcoef(smooth(self.leadingFreqsOverTime,window_len=30), smooth(f(squeezedTime),window_len=30))[0][1]


# names = ['kurtcan1','kurtcan2','kurtcan3','vural1','vural2','vural3','alimert1','alimert2','alimert3','caner1','caner2','caner3','anten1','anten2','anten3','batikan1','batikan2','batikan3','diedon1','diedon2','diedon3','alakasiz1','alakasiz2','alakasiz3','alakasiz4','alakasiz5']
#
# results = {}
# for el in list(itertools.combinations(names, 2)):
#     whistle1 = Whistle('whistles/' + el[0] +'.wav')
#     whistle2 = Whistle('whistles/' + el[1] +'.wav')
#     results[el[0] + ' - ' + el[1] ] = whistle1.getCorrWith(whistle2)
#
# sortedResults = sorted(results.items(), key=operator.itemgetter(1),reverse=True)
# for el in sortedResults:
#     print(el[0] + ' ==> ' + str(el[1]))

# whistle1 = Whistle('vuraltest1.wav')
# whistle2 = Whistle('vuraltest2.wav')
# print(whistle1.getCorrWith(whistle2))

names = ['kurtcan', 'vural', 'alimert', 'caner', 'anten', 'batikan', 'diedon']

for name in names:
    whistle1 = Whistle('whistles/' + name + '1.wav')
    whistle2 = Whistle('whistles/' + name + '2.wav')
    whistle3 = Whistle('whistles/' + name + '3.wav')
    corr12 = whistle1.getCorrWith(whistle2)
    corr13 = whistle1.getCorrWith(whistle3)
    corr23 = whistle2.getCorrWith(whistle3)
    print(name + ' ==> ' + str((corr12 + corr13 + corr23)/3))