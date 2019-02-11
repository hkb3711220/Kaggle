import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from load_data import *
import seaborn as sns

#device - The specifications for the device used to access the Store.

device = load_data(target='train', rebuild=False)._read_csv('device', chunksize=600)
geoNetwork = load_data(target='train', rebuild=False)._read_csv('geoNetwork', chunksize=600)
totals = load_data(target='train', rebuild=False)._read_csv('totals', chunksize=600)

data = pd.concat([device, geoNetwork, totals], axis=1).replace('not available in demo dataset', np.nan)
#data = data.dropna(axis=1)
row_data = data.dropna(axis=0)
data = data.dropna(axis=1)

browser_vc = data['browser'].value_counts(sort=True)
fig, axes = plt.subplots(nrows=3, ncols=2)
sns.barplot(x=browser_vc.index, y=browser_vc.values, ax=axes[0,0])
sns.countplot(x='browser', hue='operatingSystem', data=data, palette="Set1", ax=axes[0,1])
sns.countplot(x='browser', hue='isMobile', data=data, palette="Set1",ax=axes[1,0])
sns.countplot(x='browser', hue='deviceCategory', data=data, palette="Set1",ax=axes[1,1])
sns.countplot(x='browser', hue='continent', data=data,palette="Set1", ax=axes[2,0])
sns.countplot(x='operatingSystem', hue='continent', data=data, palette="Set1", ax=axes[2,1])

sns.catplot(x='browser', y='pageviews',
            row='continent', hue='isMobile', kind='violin',
            height=2, aspect=3, palette="Set3", data=data,
            dodge=True, cut=0, bw=.2)
#sessionQualityDim: An estimate of how close a particular session was to transacting,
#ranging from 1 to 100,calculated for each session.

sns.catplot(y='sessionQualityDim', x='pageviews', hue='isMobile', kind='point',
            height=2, aspect=3, palette="Set2", data=data,
            dodge=True, cut=0, bw=.2)

sns.catplot(y='sessionQualityDim', x='browser', hue='isMobile', kind='point',
            height=2, aspect=3, palette="Set1", data=data,
            dodge=True, cut=0, bw=.2)

plt.show()
