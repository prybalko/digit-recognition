import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("data/train-less.csv", sep=",")

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(dataset.iloc[i][1:].values.reshape([28, 28]))
plt.show()
