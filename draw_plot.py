import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_data = pd.read_csv('data_error.csv')
step = train_data['epoch'].values
validation = train_data['train_loss'].values


plt.plot(step,validation)
plt.show()
