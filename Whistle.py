import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from scipy.interpolate import interp1d
import itertools
import operator
from sklearn.metrics import f1_score

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
        if len(snd.shape) > 1:
            numOfPoints, numOfChannels = snd.shape
        else:
            numOfChannels = 1
        if numOfChannels == 2:
            monoChannelSound = (snd[:, 0] + snd[:, 1]) / 2
        else:
            monoChannelSound = snd
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

# names = ['kurtcan', 'vural', 'alimert', 'caner', 'anten', 'batikan', 'diedon']
#
# for name in names:
#     whistle1 = Whistle('whistles/' + name + '1.wav')
#     whistle2 = Whistle('whistles/' + name + '2.wav')
#     whistle3 = Whistle('whistles/' + name + '3.wav')
#     corr12 = whistle1.getCorrWith(whistle2)
#     corr13 = whistle1.getCorrWith(whistle3)
#     corr23 = whistle2.getCorrWith(whistle3)
#     print(name + ' ==> ' + str((corr12 + corr13 + corr23)/3))

# names = ['kurtcan', 'vural', 'alimert', 'caner', 'anten', 'batikan', 'diedon']
# for el in list(itertools.combinations(names, 2)):
#     totalCorr = 0
#     for i in range(1,4):
#         for j in range(1,4):
#             whistle1 = Whistle('whistles/' + el[0] + str(i) + '.wav')
#             whistle2 = Whistle('whistles/' + el[1] + str(j) + '.wav')
#             totalCorr += whistle1.getCorrWith(whistle2)
#     print(el[0] + '-' + el[1] + '==> ' + str(totalCorr/9))

# names = ['kurtcan', 'vural', 'alimert', 'caner', 'anten', 'batikan', 'diedon']
# corrs = np.array([])
# for name in names:
#     for i in range(1,4):
#         for j in range(1,6):
#             whistle1 = Whistle('whistles/' + name + str(i) + '.wav')
#             whistle2 = Whistle('whistles/' + 'alakasiz' + str(j) + '.wav')
#             corrs = np.append(corrs, [whistle1.getCorrWith(whistle2)])
# print('Average of false matches ==> ' + str(np.average(corrs)))
# print('Std of false matches ==> ' + str(np.std(corrs)))

# names = ['kurtcan1','kurtcan2','kurtcan3','vural1','vural2','vural3','alimert1','alimert2','alimert3','caner1','caner2','caner3','anten1','anten2','anten3','batikan1','batikan2','batikan3','diedon1','diedon2','diedon3','alakasiz1','alakasiz2','alakasiz3','alakasiz4','alakasiz5']
#
# results = {}
# corrs = np.array([])
# real = np.array([])
# pred = np.array([])
# for el in list(itertools.combinations(names, 2)):
#     whistle1 = Whistle('whistles/' + el[0] +'.wav')
#     whistle2 = Whistle('whistles/' + el[1] +'.wav')
#     resultCorr = whistle1.getCorrWith(whistle2)
#     corrs = np.append(corrs, [resultCorr])
#     if resultCorr > 0.1:
#         pred = np.append(pred, [1])
#     else:
#         pred = np.append(pred, [0])
#     if el[0][:-1] == 'alakasiz' or el[1][:-1] == 'alakasiz':
#         real = np.append(real, [0])
#     else:
#         real = np.append(real, [1])
# print(f1_score(real,pred))

# names = ['kurtcan1','kurtcan2','kurtcan3','vural1','vural2','vural3','alimert1','alimert2','alimert3','caner1','caner2','caner3','anten1','anten2','anten3','batikan1','batikan2','batikan3','diedon1','diedon2','diedon3','alakasiz1','alakasiz2','alakasiz3','alakasiz4','alakasiz5']
#
# results = {}
# corrs = np.array([])
# real = np.array([])
# pred = np.array([])
# for el in list(itertools.combinations(names, 2)):
#     whistle1 = Whistle('whistles/' + el[0] +'.wav')
#     whistle2 = Whistle('whistles/' + el[1] +'.wav')
#     resultCorr = whistle1.getCorrWith(whistle2)
#     corrs = np.append(corrs, [resultCorr])
#     if resultCorr > 0.35:
#         pred = np.append(pred, [1])
#     else:
#         pred = np.append(pred, [0])
#     if el[0][:-1] != 'alakasiz' and el[1][:-1] != 'alakasiz' and el[0][:-1] == el[1][:-1]:
#         real = np.append(real, [1])
#     else:
#         real = np.append(real, [0])
# print(f1_score(real,pred))


whistle1 = Whistle('shortWhistles/anne-diedon2.wav')
whistle2 = Whistle('shortWhistles/anne-diedon3.wav')
print(whistle1.getCorrWith(whistle2))
