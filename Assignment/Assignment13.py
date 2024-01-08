import numpy as np
import matplotlib.pyplot as plt


## Implement the formula given above in the following function template. 
## The input x array already given for you.

def univariate_gaussian_pdf(x, mean, variance):

    result = 1/np.sqrt(2*np.pi*variance)*np.exp(-((x-mean)**2)/2*(variance))
    return result
    
    

mean = 0
stddev = 1
x = np.arange(-5, 5, 0.01)
y = univariate_gaussian_pdf(x, mean, stddev**2)
#plt.plot(x,y)
plt.plot(sample)
plt.hist(x,normed=True,bins=np.arange(-7, 7, 0.5))
plt.xlabel("x")
plt.ylabel("PDF")
plt.show()



sample_size = 1000

rng = np.random.RandomState(123)
sample = rng.normal(loc=0.0, scale=1.0, size=sample_size)

# In the following function template, please implement the empirical PDF
# 1. Sort the sample
# 2. Calculate the mean and variance of the sample
# 3. Calculate the PDF of the sample using the function univariate_gaussian_pdf
# 4. Return the sorted sample and the PDF of the sample
def empirical_pdf(sample):
    ### your code goes here
    sample_sorted = np.sort(sample)
    sample_mean = np.mean(sample)
    sample_variance = np.var(sample, ddof=1)
    sample_pdf = univariate_gaussian_pdf(sample_sorted, 
                                     sample_mean, 
                                     sample_variance)
    return sample_sorted, sample_pdf
    ### your code goes here

sample_sorted, sample_pdf = empirical_pdf(sample)
# Plot the empirical PDF and the theoretical PDF, label the graph with name 'empirical'
# Hint: x = sample_sorted, y = sample_pdf
# On the same graph, plot the theoretical PDF with x = x, y = y, label the graph with name 'theoretical'


### your code goes here
sample_sorted, sample_pdf = empirical_pdf(sample)
plt.plot(sample_sorted, sample_pdf, label='empirical')
plt.plot(x, y, label='theoretical')
plt.ylabel('PDF')
plt.xlabel('x')
plt.legend()
plt.show()

### your code goes here