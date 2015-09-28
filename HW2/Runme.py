
# coding: utf-8

# In[29]:


# import libraries
# get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import deque

#### Use Hawkins CUSUM Method described in http://rmgsc.cr.usgs.gov/outgoing/threshold_articles/Hawkins_Zamba2005b.pdf
#### and in Douglas C. Montgomery Statistical Quality Control 6th Edition Page 400-Page 417 Chapter Time-Weighted Control Charts
#### Tunable Parameters:
#### k : Allowance value, a threshold that signals a possible shift starts at the current position
#### H : If CUSUM value goes beyond this value, conclude a shift has occured and the shift starts at the last point
####  whose CUSUM is larger than k Allowance value
#### Seperate k and H values are used for mean and var shift detection


def CUSUMDetect(data):
    
    cusums_mean_up = []   # Ci array for mean upper shift detection
    N_mean_up = [] # N+ array for mean upper shift detection
    n_mean_up = 0 # N1 = 0
    cusum_mean_up = 0   # C1 = 0 for mean upper shift detection
    cusums_mean_lo = []   # Ci array for mean lower shift detection
    N_mean_lo = [] # N+ array for mean lower shift detection
    n_mean_lo = 0 # N1 = 0
    cusum_mean_lo = 0   # C1 = 0 for mean lower shift detection
    
    cusums_var_up = [] # Ci array for var upper shift detection
    N_var_up = [] # N+ arrary for var upper shift detection
    n_var_up = 0 # N1 = 0
    cusum_var_up = 0 # C1 = 0 for var upper shift detection
    cusums_var_lo = [] # Ci array for var lower shift detection
    N_var_lo = [] # N+ arrary for var lower shift detection
    n_var_lo = 0 # N1 = 0
    cusum_var_lo = 0 # C1 = 0 for var lower shift detection
    
    n = 50 # The first n samples form the baseline
    i = 1 # The index looping through
    mean = np.mean(data[:n]) # mean of baseline
    sd = np.std(data[:n]) # std of baseline
    k = 2 * sd  # Allowance value for mean shift detection
    H = 12 * sd # Decision interval for mean shift detection
    k_v = 1 * sd # Allowance value for var shift detection
    if mean/sd <= 2:
        H_v = 1.5 * mean/sd * sd # Decision interval for var shift detection that mean and std are close 1.5
    else:
        H_v = 5* mean/sd *sd # Decision interval for var shift detection that std is much smaller than mean
    #print ("mean of 50 samples = " + str(mean))
    #print ("std of 50 samples = " + str(sd))
    #print ("Allowance value of mean = " + str(k))
    #print ("Decision interval of mean = " + str(H))
    #print ("Allowance value of variance = " + str(k_v))
    #print ("Decision interval of variance = " + str(H_v))
    
   
    
    #for x in data[n+1:]: # Start building CUSUM tabular from the n+1 point
    for x in data: # Start building CUSUM tabular 
        
        ## Tabular CUSUM for mean shift detection
        # mean upper shift CUSUM tabular generation
        cusum_mean_up = max(0, x - (mean + k) + cusum_mean_up) # Ci = max(0, xi-(mean+k)+Ci-1) for mean shift detection
        
        if cusum_mean_up == 0: # if cusum value is 0 
             n_mean_up = 0    # not start recording a possible shift change, set index Number to 0
        else:
             n_mean_up = n_mean_up + 1 # else if cusum value is not 0 
        N_mean_up.append(n_mean_up) # start recording a possible shift change, increment index Number by 1
            
        cusums_mean_up.append(cusum_mean_up) # Append CUSUM
        
        # mean lower shift CUSUM tabular generation
        cusum_mean_lo = max(0, (mean - k) - x + cusum_mean_lo) # Ci = max(0, (mean-k)-xi+Ci-1) for mean shift detection
        
        if cusum_mean_lo == 0: # if cusum value is 0 
             n_mean_lo = 0     # not start recording a possible shift change, set index Number to 0
        else:
             n_mean_lo = n_mean_lo + 1 # else if cusum value is not 0 
        N_mean_lo.append(n_mean_lo)   # start recording a possible shift change, increment index Number by 1
             
        cusums_mean_lo.append(cusum_mean_lo) # Append CUSUM 
        
        ## Tabular CUSUM for var shift detection
        ## Ci  for var shift detection,Hawkins metod: yi = (xi - mean)/std, vi = (sqrt(abs(yi)) - 0.822)/0.349
        ## Si+ = max(0,vi-k+Si-1) , Si- = max(0,-k-vi+Si-1)
        
        # var upper shift CUSUM tabular generation
        y = (x - mean)/sd
        if mean/sd <= 2:
           v = ((np.sqrt(abs(y))) - 0.822) / 0.01 # v for mean and std are very close like posVarup 60 file
        else:
           v = ((np.sqrt(abs(y))) - 0.822) / 0.03  #file for std is much smaller than mean like posVarup70 file
        
        cusum_var_up = max(0, v - k_v + cusum_var_up)
        
        if cusum_var_up == 0: # if cusum value is 0 
             n_var_up = 0     # not start recording a possible shift change, set index Number to 0
        else:
             n_var_up = n_var_up + 1 # else if cusum value is not 0 
        N_var_up.append(n_var_up)   # start recording a possible shift change, increment index Number by 1
        
        cusums_var_up.append(cusum_var_up) # Append CUSUM 
        
        # var lower shift CUSUM tabular generation
        # cusum_var_lo = max(0,-v - k_v + cusum_var_up)
        cusum_var_lo = abs(-v - k_v + cusum_var_up)
        
        if cusum_var_lo == 0: # if cusum value is 0 
             n_var_lo = 0     # not start recording a possible shift change, set index Number to 0
        else:
             n_var_lo = n_var_lo + 1 # else if cusum value is not 0 
        N_var_lo.append(n_var_lo)   # start recording a possible shift change, increment index Number by 1
        
        cusums_var_lo.append(cusum_var_lo) # Append CUSUM 
        
        
        i = i + 1
        ## Decide if shift in mean or var is detected
        if cusum_mean_up > H:
           
            c_i = i - (n_mean_up - 1) # The position where change right starts
            if c_i > 50:
                print("Mean shifted up at position " + str(c_i))

                #print("CUSUM = " + str(cusums_mean_up))
                '''
                plt.plot(cusums_mean_up)
                plt.plot([c_i, c_i], [data[c_i]+1, 0], 'r--', color='y')  # Plot the change line
                plt.text(c_i, data[c_i] - 0.05, 'Change position',color='y') # Label out change position
                plt.title("CUSUM of MEAN UP " + str(c_i) + " observations")
                plt.show()
                '''
                return c_i
        
        if cusum_mean_lo > H:
            
            c_i = i - (n_mean_lo - 1) # The position where change right starts
            if c_i > 50:
                print("Mean shifted down at position " + str(c_i))
                #print("CUSUM = " + str(cusums_mean_lo))
                '''
                plt.plot(cusums_mean_lo)
                plt.plot([c_i, c_i], [data[c_i]+1, 0], 'r--', color='y')  # Plot the change line
                plt.text(c_i, data[c_i] - 0.05, 'Change position',color='y') # Label out change position
                plt.title("CUSUM of MEAN DOWN" + str(c_i) + " observations")
                plt.show()
                '''
                return c_i
        
        
        if cusum_var_up > H_v:
            #c_i = i + n + 1 - (n_var_up - 1) # The position where change right starts
            c_i = i  - (n_var_up - 1)
            if c_i > 50:
                print("Variance shifted up at position " + str(c_i))
                #print("CUSUM = " + str(cusums_var_up))
                '''
                plt.plot(cusums_var_up)
                plt.plot([c_i, c_i], [data[c_i]+1, 0], 'r--', color='y')  # Plot the change line
                plt.text(c_i, data[c_i] - 0.05, 'Change position',color='y') # Label out change position
                plt.title("CUSUM of VAR up" + str(c_i) + " observations")
                plt.show()
                '''
                return c_i
        
        if cusum_var_lo > H_v:
            #c_i = i + n + 1 - (n_var_lo - 1) # The position where change right starts
            c_i = i  - (n_var_lo - 1)
            if c_i > 50:
                print("Variance shifted down at position " + str(c_i))
                #print("CUSUM = " + str(cusums_var_lo))
                '''
                plt.plot(cusums_var_lo)
                plt.plot([c_i, c_i], [data[c_i]+1, 0], 'r--', color='y')  # Plot the change line
                plt.text(c_i, data[c_i] - 0.05, 'Change position',color='y') # Label out change position
                plt.title("CUSUM of VAR down" + str(c_i) + " observations")
                plt.show()
                '''
                return c_i
        
       
    print("No Change detected using CUSUM method")

    return -1

#### Function to convert attributes data to continous data
#### Assign char a,b,c to uniformly distributed value seperately for all data points first
#### Then, group data samples into sub group first,with each group has n data points,calculate the average of each subgroup
#### Therefore we get a new continous data array named sample
#### Plot the sample and a shift can be seen obviously
#### To-do: Experiment using p chart, or Shewart chart or CUSUM or other methods to detect the change position
#### and in Douglas C. Montgomery Statistical Quality Control 6th Edition Page 400-Page 417 Chapter Time-Weighted Control Charts
#### Tunable Parameters:
#### n : subgroup sample size n 
#### t: The first t data points be selected as baseline


def AttrToCont(lines):
    
    np.random.seed(123456789)

    data = [] # Initiate an array "data" to hold original attributes data 
    for x in lines:
        if x == 'a':    #  # Convert char a to a random number from 0 to 1 and append to data array
            #data.append(0) 
            data.append(np.random.uniform(0,1,1))
        if x == 'b':     # Convert char b to a random number from 1 to 2 and append to data array
            #data.append(1)
            data.append(np.random.uniform(1,2,1))
        if x == 'c':    # Convert char c to a random number from 2 to 3 and append to data array
            #data.append(2)
            data.append(np.random.uniform(2,3,1))
        

    n = 14 # Tunable parameter to group each n+1 data points into a new sample point
    m = len(data)/(n+1) # Number of samples converted from original data
    sample = [] # A new array to hold continous data converted from attributes data

    ## Fill in the new sample data array converted from original attributes data
    for i in range(int(m)):
        if i == 0:
            sample.append(sum(data[:n])/(n+1))   # Append the average values of each n+1 data points
        else:
            sample.append(sum(data[i*n+1:(i+1)*n])/(n+1))  # Append the average values of each n+1 data points

    #print("Converted variable values are" + str(sample))
    
    return sample

#### Shewart control chart method is used, used first t samples to set up and calculate baseline limits
#### including mean, UCL,LCL,UWL,LWL, and then scan next 5 points in a slide window
### to compare the points relationship and trend against  baseline limits
### if 3 consecutive points outside UCL/LCL, detect signal of change
#### or if 5 consecutive points outside UWL/LWL, detect signal of change


def shewartDetect(sample,s):

    t = 4 # Choose the first t+1 numbers of sample points as baseline

    ## Calculate control limits using P chart method
    mean = np.average(sample[:t]) # Get the mean of baseline data
    sd = np.std(sample[:t])
    #UCL = mean + 3*np.sqrt(mean*(1-mean)/(n+1)) # Upper control limit in P chart, does not work for negtriple, tripledouble
    #LCL = mean - 3*np.sqrt(mean*(1-mean)/(n+1)) # Lower control limit in P chart,does not work for negtriple, tripledouble
    UCL = mean + 4*sd # Upper control limit in Shewart method
    LCL = mean - 3*sd # Lower control limit in Shewart method
    UWL = mean + 2*sd # Upper control limit in Shewart method
    LWL = mean - 2*sd # Lower control limit in Shewart method

 
    n = 0 # index starts from 0 which means not a point outsidte control threshold
    m = 0 # index starts from 0 which means not a point outsidte warning threshold
    ## Detect if there is concept change occurs after the baseline sample points
    for i in range(t+1,len(sample)):
        if (sample[i] >= UCL) | (sample[i] <= LCL): # If there is a point outside threshold
            n = n + 1
            #N.append(n)
            if n == 3: # If it is the consecutive 3rd point that is outside threshold
                
                #print("That corresponds to the "+str(i-2)+"th position on the converted data graph below")
                #print("The change occur sample value is"+str(sample[i-3]))
                '''
                plt.plot(sample[:i]) # Plot the converted variable smaple data
                plt.plot([0, len(sample)], [UCL, UCL], 'r--', color='r')  # Plot the Upper Control Limit
                plt.plot([0, len(sample)], [LCL, LCL], 'r--', color='r') # Plot the Lower Control LIMIT
                plt.text(1, UCL - 0.05, 'UCL',color='r') # Label out UCL 
                plt.text(1, LCL + 0.05, 'LCL',color='r') # Label out LCL 
                plt.plot([i-3, i-3], [sample[i]+1, 0], 'r--', color='y')  # Plot the change line
                plt.text(i-3, sample[i] - 0.05, 'Change position',color='y') # Label out change position
                plt.plot([0, len(sample)], [UWL, UWL], 'r--', color='r')  # Plot the Upper Warning Limit
                plt.plot([0, len(sample)], [LWL, LWL], 'r--', color='r') # Plot the Lower Warning LIMIT
                plt.text(1, UWL - 0.05, 'UWL',color='r') # Label out UWL 
                plt.text(1, LWL + 0.05, 'LWL',color='r') # Label out LWL 
                plt.title("Change Ocurred because 3 consective points outsidte UCL OR LCL")
                plt.show()
                '''
                if s == True:
                    print("Concept change occured at position " + str((i-3)*(n+1)*3) + " of original attributes data")
                    return (i - 3)*(n+1)*3
                elif s == False:
                    print("Concept change occured at position " + str(i-3))
                    return i - 3
        elif (sample[i] >= UWL) | (sample[i] <= LWL): # If there is a point outside warning threshold:
            m = m + 1
            if m == 5: # If it is the consecutive 3rd point that is outside threshold
                
                #print("That corresponds to the "+str(i-2)+"th position on the converted data graph below")
                #print("The change occur sample value is"+str(sample[i-4]))
                '''
                plt.plot(sample[:i]) # Plot the converted variable smaple data
                plt.plot([0, len(sample)], [UCL, UCL], 'r--', color='r')  # Plot the Upper Control Limit
                plt.plot([0, len(sample)], [LCL, LCL], 'r--', color='r') # Plot the Lower Control LIMIT
                plt.plot([i-4, i-4], [sample[i]+1, 0], 'r--', color='y')  # Plot the change line
                plt.text(i-4, sample[i] - 0.05, 'Change position',color='y') # Label out change position
                plt.text(1, UCL - 0.05, 'UCL',color='r') # Label out UCL 
                plt.text(1, LCL + 0.05, 'LCL',color='r') # Label out LCL
                plt.plot([0, len(sample)], [UWL, UWL], 'r--', color='r')  # Plot the Upper Warning Limit
                plt.plot([0, len(sample)], [LWL, LWL], 'r--', color='r') # Plot the Lower Warning LIMIT
                plt.text(1, UWL - 0.05, 'UWL',color='r') # Label out UWL 
                plt.text(1, LWL + 0.05, 'LWL',color='r') # Label out LWL 
                plt.title("Change Ocurred because 5 consecutive points outside UWL OR LWL")
                plt.show()
                '''
                if s == True:
                    print("Concept change occured at position " + str((i-4)*(n+1)*3) + " of original attributes data")
                    return (i - 4)*(n+1)*3
                elif s == False:
                    print("Concept change occured at position " + str(i-4))
                    return i - 4
               
        else:
            n = 0
            m = 0
    
         
    
    print("No Change detected using Shewart method")
    '''
    plt.plot(sample) # Plot the converted variable smaple data
    plt.plot([0, len(sample)], [UCL, UCL], 'r--', color='r')  # Plot the Upper Control Limit
    plt.plot([0, len(sample)], [LCL, LCL], 'r--', color='r') # Plot the Lower Control LIMIT
    plt.text(1, UCL - 0.05, 'UCL',color='r') # Label out UCL 
    plt.text(1, LCL + 0.05, 'LCL',color='r') # Label out LCL 
    plt.plot([0, len(sample)], [UWL, UWL], 'r--', color='r')  # Plot the Upper Control Limit
    plt.plot([0, len(sample)], [LWL, LWL], 'r--', color='r') # Plot the Lower Control LIMIT
    plt.text(1, UWL - 0.05, 'UWL',color='r') # Label out UCL 
    plt.text(1, LWL + 0.05, 'LWL',color='r') # Label out LCL 
    plt.title("Converted Variable Plot from Original Attributes Data")
    plt.show()
    '''

    return -1

#### Use Kolmogorov-Smirnov Test described in http://www.physics.csbsju.edu/stats/KS-test.html
#### Tunable Parameters:
#### a : Significant level (0.05 or 0.01) to reject the null hypothesis that there is no change (default is 0.01)
#### w_size : Window size (default is 15 observations)
#### overlap : set window size overlapping
#### Usage example: data = np.loadtxt("posShiftUpMean_70.txt", delimiter=",") KSDetect(data)

def KSdetect(data, w_size=20, overlap=0, a=0.01):

    # loop to detect change
    for i in range(len(data)):
        if i - w_size >= w_size:
            W1 = data[i - w_size: i] # set current window from current location minus w_size
            W0 = data[i - (w_size * 2) + overlap:i - w_size + overlap] # set reference window

            # find komogorov-smirnov
            ks = stats.ks_2samp(W0, W1)
            p = ks[1]

            # reject H0 and conclude that there is a change at n
            if p < a:
                print("Change of distribution detected at " + str(i) + " with p-value = " + str(p))
                return i
        else:
            pass

    print("No change detected using KS method")
    return -1

#### Use Chi-Square. Specify window size and alpha
#### Tunable Parameters:
#### a : Significant level (0.05 or 0.01) to reject the null hypothesis that there is no change (default is 0.01)
#### w_size : Window size
#### overlap : set window size overlapping

def CHdetect(lines, w_size=50, overlap=40, a=0.01):

    data = [] # Initiate an array "data" to hold original attributes data 
    for x in lines:
        if x == 'a':    # Convert char a to number 0 and append to data array
            data.append(0) 
        if x == 'b':    # Convert char b to number 1 and append to data array
            data.append(1)
        if x == 'c':    # Convert char c to number 2 and append to data array
            data.append(2)     


    # loop to detect change
    for i in range(len(data)):
        if i - w_size >= w_size:
            W1 = data[i - w_size: i] # set current window from current location minus w_size
            W0 = data[i - (w_size * 2) + overlap:i - w_size + overlap] # set reference window

            # find expected and observed value of a, b, c
            if W0.count(2) == 0 and W1.count(2) == 0:
                expected = [W0.count(0), W0.count(1)]
                observed = [W1.count(0), W1.count(1)]
            else:
                expected = [W0.count(0), W0.count(1), W0.count(2)]
                observed = [W1.count(0), W1.count(1), W0.count(2)]

            # find chi square
            if 0 in expected:
                print("No Chi Square test provided")
                return -1
            else:
                chi = stats.chisquare(f_obs=observed, f_exp=expected)
                p = chi[1]

            # reject H0 and conclude that there is a change at n
            if p < a:
                print("Change detected at " + str(i) + " with p-value = " + str(p))
                return i
        else:
            pass
    print("No change detected using Chi Square method")
    return -1

## Function used to detect concept change for continous variable data stream
### using a combination of methods defined above

def shiftDetectCont(data):

    s = False
    shew = shewartDetect(data,s)  # Use Shewart Control Chart to detect any concept change first
    if shew == -1: # If no change detected by Shewart control chart      
        cusum = CUSUMDetect(data)  # Then use CUSUM to detect any concept change first
        
        if cusum == -1: # If no change detected by CUSUM
            ks = KSdetect(data) # Then use Kolmogorov-Smirnov
            if ks == -1:
                return -1
            else:
                return ks
        else:
            return cusum
    else:
        return shew

## Function used to detect concept change for continous variable data stream
### using a combination of methods defined above

def shiftDetectAtt(data):
    
    sample = AttrToCont(data) # Convert attribute data to continous variable data, and store them in sample variable
    s = True
    sh = shewartDetect(sample,s) # Then apply continous data detection function on converted continous data
    if sh == -1:
        ch = CHdetect(data)
        return ch
    else:
        return sh
    
    #return shewartDetect(sample)

## Handling missing, invalid data


## Definition of shiftDetect(data)
### Read all the provided and generated sample files start with neg or pos, end with .txt
## Auto convert attribute data to continous data 
### Append results to array and write to Dingchao_and_Pipat.txt

def shiftDetect(path):
    
    ## Provide a path, and randomly pick 10 txt files to read
    import os # import os module
    import random # import random module
    allfiles = [] # Array to hold all txt files names 
    for file in os.listdir(path): # For all the files in the current work directory
        if file.endswith(".txt") and (file.startswith("pos") or file.startswith("neg")):  # If it is a txt file 
            allfiles.append(file) # Append that file name to allfiles array
    #random.shuffle(allfiles) # Randomly shuffle allfiles
    #n = 13 # Tunable parameter, choose the firt n shuffled files to read
    #files = allfiles[:n] # Choose the first n files
    files = allfiles
    output = [] # Output Array of file names
    output2= [] # Output array of values
    for i in range(0,len(files)): # Detect concept change for all the randomly selected files
        print("Reading " + files[i])
        with open(files[i]) as f: # Open file as f
            lines = f.read().splitlines()  # Split file line by line and put lines in to var "lines"
        f.close() # Close file
        output.append(files[i])

        for i in range(len(lines)): # if missing data, assign the previous one
            if not lines[i]:
                lines[i] = lines[i - 1] 

        try:
            data = [float(i) for i in lines] # For variable data stream,convert data string to float if convertable
            #shiftDetectCont(data) # Apply continous detection function to see if concept changes
            output2.append(shiftDetectCont(data))
        except: 
            data = lines # Otherwise for attribute data, no convert needed
            #shiftDetectAtt(data) # Apply attributes detection function to see if concept changes
            output2.append(shiftDetectAtt(data))
    

    list_a = list(zip(output, output2)) # Return a tuple of filename and its value
    
    with open('Dingchao_and_Pipat.txt', "w") as f:
        f.write("Dingchao Zhang\tPipat Thontirawong\n")
        for x, y in list_a:
            f.write("{}\t{}\n".format(x, y))
        f.close()
    print(list_a)
    return

# randomly generate test data
import numpy as np

# sample with mean change
def sampleMean(i):
    have_change = np.random.choice([True, False])
    val = np.random.randint(0, 100, size=2)
    var = np.random.randint(0, 10, size=1)

    if have_change:
        position_change = np.random.randint(51, 1000)
        data1 = np.random.normal(loc=val[0], scale=var, size=position_change)
        data2 = np.random.normal(loc=val[1], scale=var, size=1000 - position_change)
        data = np.append(data1, data2)
        if val[1] > val[0]:
            filename = 'posShiftUpMean_' + str(position_change) + str(i) +  '.txt'
        else:
            filename =  'posShiftDownMean_' + str(position_change) + str(i) + '.txt'
        
    else:
        position_change = -1
        data = np.random.normal(loc=val[0], scale=var, size=1000)
        filename = 'negMean_'+str(i)+ '.txt'

    np.savetxt(filename, np.round(data, 3), fmt='%.3f')
    return "write sample file " + filename

# sample with variance change
def sampleVariance(i):
    have_change = np.random.choice([True, False])
    val = np.random.randint(0, 100, size=1)
    var = np.random.randint(0, 10, size=2)

    if have_change:
        position_change = np.random.randint(51, 1000)
        data1 = np.random.normal(loc=val, scale=var[0], size=position_change)
        data2 = np.random.normal(loc=val, scale=var[1], size=1000 - position_change)
        data = np.append(data1, data2)
        if var[1] > var[0]:
            filename = 'posShiftUpVar_' + str(position_change) + str(i) + '.txt'
        else:
            filename = 'posShiftDownVar_' + str(position_change) + str(i) +  '.txt'

    else:
        position_change = -1
        data = np.random.normal(loc=val, scale=var[0], size=1000)
        filename = 'negVar_' + str(i)  +  '.txt'

    np.savetxt(filename, np.round(data, 3), fmt='%.3f')
    return "write sample file " + filename

# sample with change in probability of binary
def sampleBinary(i):
    have_change = np.random.choice([True, False])
    p = np.random.uniform(0, 1, size=2)
    choices = ['a', 'b']

    if have_change:
        position_change = np.random.randint(51, 1000)
        data1 = np.random.choice(choices, position_change, p=[p[0], 1 - p[0]])
        data2 = np.random.choice(choices, 1000 - position_change, p=[p[1], 1 - p[1]])
        data = np.append(data1, data2)
        filename = 'posBinary_' + str(position_change) + str(i) + '.txt'

    else:
        position_change = -1
        data = np.random.choice(choices, 1000, p=[p[0], 1 - p[0]])
        filename = 'negBinary_' + str(i) + '.txt' 

    lst = data.tolist()
    with open(filename, "w") as f:
        for x in lst:
            f.write(x + "\n")
        f.close()
    return "write sample file " + filename

# sample with change in probability of binary
def sampleTriple(i):
    have_change = np.random.choice([True, False])
    choices = ['a','b','c']
    p = np.random.dirichlet(np.ones(10),size=1)
    p_0 = np.sum(p[0][:3])
    p_1 = np.sum(p[0][4:5])
    p_2 = 1- p_0 - p_1 

    if have_change:
        position_change = np.random.randint(51, 100)
        data1 = np.random.choice(choices, position_change, p=[p_0,p_1,p_2])
        data2 = np.random.choice(choices, 100 - position_change, p=[p_2,p_0,p_1])
        data = np.append(data1, data2)
        filename = 'posTriple_' + str(position_change) +str(i) +  '.txt'
        #print("change" + str(data))
    else:
        position_change = -1
        data = np.random.choice(choices, 100, p=[p_0,p_1,p_2])
        filename = 'negTriple_' +str(i) +'.txt'
        #print(" no change" + str(data))

    
    
    lst = data.tolist()
    with open(filename, "w") as f:
        for x in lst:
            f.write(x + "\n")
        f.close()
    return "write sample file " + filename

## Detect the current working directory which contains the data files
import os
shiftDetect(os.getcwd())


# In[ ]:



