
import numpy as np

class selectSamples():                              
    def __init__(self, timeSerie, predictionHorizonMax, window):
        self.timeSerie = timeSerie
        self.predictionHorizonMax = predictionHorizonMax
        self.window = window

    def SerieMatriz(self):
        timeSerie = np.squeeze(self.timeSerie)
        self.X = np.zeros([len(self.timeSerie)-self.predictionHorizonMax-self.window+1, self.window])
        self.y = np.zeros([len(self.timeSerie)-self.predictionHorizonMax-self.window+1, self.predictionHorizonMax])
        for i in range(self.X.shape[0]):
            self.X[i,:] = timeSerie[i:i+self.window]
            self.y[i,:] = timeSerie[i+self.window: i+self.window+self.predictionHorizonMax]
        self.y = np.squeeze(self.y)        

        return self.X, self.y


