from matplotlib import pyplot as plt
import numpy as np


class Exploration():
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def plot(self):
        xs_train = np.matrix(self.train_data[0]).T.A
        ys_train = np.matrix(self.train_data[1]).T.A

        xs_train_sorted = np.sort(xs_train)
        ys_train_sorted = np.sort(ys_train)

        px = 1/plt.rcParams['figure.dpi']  
        __fig = plt.figure(figsize=(800*px, 600*px))
        plt.scatter(xs_train_sorted, ys_train_sorted, color='b', s=1, alpha=0.5)
        plt.title('Training Data (n=' + str(len(xs_train)) + ')')
        plt.ylabel('y (Output)')
        plt.xlabel('x (Input)')
        plt.text(1, 20000, ("Formula: y = (2 + Noise) * x"), bbox=dict(boxstyle = "square", facecolor = "white", alpha = 0.5))
        plt.savefig('plots/DataExploration.png')
        #plt.show()
        print("Exploration Plot saved...")
        print("")
