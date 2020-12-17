"""
Created on Sat Nov 21 17:20:49 2020

@author:Mainak Mandal, L'Aquila,2019(mm.crjx@gmail.com)
"""

import matplotlib.pyplot as plt
import numpy as np
import csv
h= []
err_abs= []
err_rmse= []

i=0
with open('order.csv', newline='\n') as csvfile:
	data = csv.reader(csvfile, delimiter=',', quotechar='|')
	for row in data:
		h.append(np.double(row[0]))
		err_abs.append(np.double(row[1]))
		err_rmse.append(np.double(row[2]))         
#print(x,y)
h=np.array(h)
err_abs=np.array(err_abs)
err_rmse =np.array(err_rmse)

#plotting
fig, ax = plt.subplots()  # Create a figure containing a single axes.
ax.plot(h, err_rmse)  # Plot some data on the axes.


#plt.title('Stepsize(h) vs absolute error')
plt.title('Stepsize(h) vs RMSE error')

plt.xlabel('h')
#plt.ylabel('Absolute error')
plt.ylabel('RMSE error')
plt.show()

