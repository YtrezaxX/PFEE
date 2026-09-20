# plot macroscopic response of the DEM simulation

import numpy
import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.rcParams.update({'font.size': 12})

folderExp = "dem_data/monotonic/maroscopic_response/"
data_dem = numpy.loadtxt(folderExp+"triax.plot-data.txt",skiprows=1,delimiter="\t")
data_steps = numpy.loadtxt(folderExp+"triax.state.txt",skiprows=1,delimiter=" ")

fig, ax = plt.subplots()
ax.plot(-1*data_dem[:,5]*100, -1*data_dem[:,11]/1000, '-', color='tab:grey', label='')
ax.plot(-1*data_steps[:,6]*100, -1*(data_steps[:,3]-data_steps[:,1])/1000, '-s', color='tab:orange', label='Stages')
ax.set_xlabel(r'vertical strain $\epsilon_1$ [-]')
ax.set_ylabel(r'deviatoric stress $q$ [kPa]')
plt.tight_layout()
plt.show()
