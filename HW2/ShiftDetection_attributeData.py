
# coding: utf-8

# In[1]:

# import libraries
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt


# # Detect if mean or var shifts for continous stream data
# Use Hawkins CUSUM Method described in http://rmgsc.cr.usgs.gov/outgoing/threshold_articles/Hawkins_Zamba2005b.pdf
# Tunable Parameters:
#  k : Allowance value, a threshold that signals a possible shift starts at the current position
#  H : If CUSUM value goes beyond this value, conclude a shift has occured and the shift starts at the last point whose CUSUM is   larger than k Allowance value

# In[52]:

#### Function to get detect concept change for attributes data
#### Assign char a,b,c to value 0,1,2 seperately for all data points first
#### Then, group data samples into sub group first,with each group has n data points,calculate the average of each subgroup
#### Therefore we get a new continous data array named sample
#### Plot the sample and a shift can be seen obviously
#### To-do: Experiment using p chart, or Shewart chart or CUSUM or other methods to detect the change position
#### and in Douglas C. Montgomery Statistical Quality Control 6th Edition Page 400-Page 417 Chapter Time-Weighted Control Charts
#### Tunable Parameters:
#### n : subgroup sample size n 
#### t: The first t data points be selected as baseline
## posTriple2_160,posTriple_175, posTripleDouble_175, negTriple,negWithRare,negBinary



with open('negBinary.txt') as f: # Open file as f
    lines = f.read().splitlines()  # Split file line by line and put lines in to var "lines"
f.close() # Close file

data = [] # Initiate an array "data" to hold original attributes data 
for x in lines:
    if x == 'a':    # Convert char a to number 0 and append to data array
        data.append(0) 
    if x == 'b':    # Convert char b to number 1 and append to data array
        data.append(1)
    if x == 'c':    # Convert char c to number 2 and append to data array
        data.append(2)
        
#k = 49 # Tunable parameter to extract baseline data
k = len(data) # Length of original data
baseline = data[:k] # Extrat the first k samples as baseline data
n = 14 # Tunable parameter to group each n+1 data points into a new sample point
m_baseline = (k+1)/(n+1) # Number of samples in baseline data
sample = [] # A new array to hold continous data converted from attributes data
#t = 
## Fill in the new sample data array converted from original attributes data
for i in range(int(m_baseline)):
    if i == 0:
        sample.append(sum(data[:n])/(n+1))   # Append the average values of each n+1 data points
    else:
        sample.append(sum(data[i*n+1:(i+1)*n])/(n+1))  # Append the average values of each n+1 data points

print("Converted variable values are" + str(sample))

## Calculate control limits using P chart method
m_baseline = sum(sample[:4])/len(sample[:4]) # Get the mean of baseline data
UCL = m_baseline + 3*np.sqrt(m_baseline*(1-m_baseline)/(n+1)) # Upper control limit
LCL = m_baseline - 3*np.sqrt(m_baseline*(1-m_baseline)/(n+1)) # Lower control limit


## Calculate control limits using Shewart chart method
m_baseline = sum(sample[:4])/len(sample[:4]) # Get the mean of baseline data
UCL = m_baseline + 3*np.sqrt(m_baseline*(1-m_baseline)/(n+1)) # Upper control limit
LCL = m_baseline - 3*np.sqrt(m_baseline*(1-m_baseline)/(n+1)) # Lower control limit

print("mean is in P chart method:" + str(m_baseline))
print("UCL is in P chart method: " + str(UCL))
print("LCL is in P chart method: " + str(LCL))

print("mean of 13 sample points is"  + str(sum(sample[:12])/12))
print("mean of 5 sample points is"  + str(sum(sample[:4])/4))
plt.plot(sample)
plt.plot([0, len(sample)], [UCL, UCL], 'r--', color='r')
plt.plot([0, len(sample)], [LCL, LCL], 'r--', color='r')
plt.text(1, UCL - 0.05, 'UCL',color='r')
plt.text(1, LCL + 0.05, 'LCL',color='r')
plt.title("Converted Variable Plot from Original Attributes Data")
plt.show()

#return


  


# In[ ]:



