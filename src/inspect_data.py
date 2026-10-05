import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/johnn/Documents/Python/Survival_Analysis_Project/data/raw/lung_cancer.xls")
print(df.columns)
# count_1 = df["karnoPH"].value_counts()
# count_2 = df["karnoPAT"].value_counts()

# # print(count_1)
# # print(count_2)

# ecog_vals = np.array(df["ecog"].values)
# karno_vals = np.array(df["karnoPH"].values)
# # X = np.concatenate((ecog_vals,karno_vals),axis=0)
# Y = np.stack((ecog_vals,karno_vals),axis=0)

# print(Y)


# print(ecog_vals.shape)
# print(karno_vals.shape)

# cov_est = np.cov(Y)
# print(np.corrcoef(ecog_vals.reshape(1,-1),karno_vals.reshape(1,-1)))
# corr = cov_est[0,1]/(cov_est[0,0]*cov_est[1,1])**0.5
# print(corr)
# plt.scatter(df["ecog"],df["karnoPH"])
# plt.show()


# total = 0
# ecog_mean = ecog_vals.mean()
# print(ecog_mean)
# karno_mean = karno_vals.mean()
# print(karno_mean)

# karno_square_diffs = [(val - karno_mean)**2 for val in karno_vals]
# print(sum(karno_square_diffs)/len(karno_square_diffs))


# for e,k in zip(ecog_vals,karno_vals):
#     total += (e - ecog_mean) * (k - karno_mean)    

# print(total/len(ecog_vals))

df_relevant = df[df["TIME"]>0]
df_relevant = df_relevant.sort_values(["TIME","ID"])
# print(df.groupby("ID")["Y"].apply(lambda x: x.max()).sum())


info = pd.DataFrame()
death_times = df[df["Y"]==1]["TIME"].value_counts()
death_times = death_times.sort_index(axis="index",ascending=True)
info["death_times"] = death_times.index
info["death_freq"] = death_times.values

print(info.head())


theta_vals = {}


