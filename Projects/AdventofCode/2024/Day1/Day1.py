# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 09:54:47 2024

@author: MR99924
"""
import pandas as pd

file_path = r"C:\Users\MR99924\.spyder-py3\workspace\Projects\AdventofCode\2024\Day1\Data1.xlsx"

df = pd.read_excel(file_path, header = None)

print(df.shape)
print(df.head())

if df.shape[1] > 0:
    list1 = df.iloc[:, 0].tolist()
    print(list1)

else:
    print("the first column doesn't exist, and Copilot is thick as pigshit)

if df.shape[1] > 1:
    list2 = df.iloc[:, 1].tolist()
    print(list2)

else:
    print("the second column doesn't exist, and Copilot is thick as pigshit)



#lst1 = df.iloc[:,0].tolist()
#lst2 = df.iloc[:,1].tolist()

#print(lst1)
#print(lst2)

