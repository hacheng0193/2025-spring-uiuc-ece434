import pandas as pd
import matplotlib.pyplot as plt
# Exercise 5 code here:
data_path = 'microphone/1.csv'

fig, axs = plt.subplots(1, 1, figsize=(10, 5))
# Read the data
micro_data = pd.read_csv(data_path)

# Calculate the correlation

for t in range(-1000, 1001):
    shifted_data = micro_data.shift(t).dropna()
    print(shifted_data.head())
    corr = shifted_data.corr(micro_data.iloc[:,1])
    if corr > max_corr:
        max_corr = corr
        best_t = t
original_corr = micro_data.iloc[:,0].corr(micro_data.iloc[:,1])
print(f'Original correlation: {original_corr}')
print(f'Best t: {best_t}, Max correlation: {max_corr}')
print('------------------------------------')

def plot_data(ax, data, title):
    ax.plot(data.index, data.iloc[:, 0], label='first microphone')
    ax.plot(data.index, data.iloc[:, 1], label='second microphone')
    ax.set_xlabel('Time')
    ax.set_ylabel('Samples')
    ax.set_title(title)
    ax.legend()
#plot_data(axs, micro_data, '1.csv')
#plt.show()
