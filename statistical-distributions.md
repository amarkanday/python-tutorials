# Statistical Distributions: Types, Use Cases, and Python Implementation

## 1. Normal (Gaussian) Distribution

### Description
The normal distribution is a continuous probability distribution that is symmetric around its mean, with data near the mean being more frequent than data far from the mean.

### Use Cases
- Heights and weights of a population
- Measurement errors
- IQ scores and other standardized test scores
- Financial returns over longer time periods
- Natural phenomena like blood pressure readings

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Generate normal distribution data
mu = 0  # mean
sigma = 1  # standard deviation
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
y = stats.norm.pdf(x, mu, sigma)

# Plot the distribution
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'r-', lw=2)
plt.title('Normal Distribution (μ=0, σ=1)')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.grid(True)

# Generate random samples from normal distribution
samples = np.random.normal(mu, sigma, 1000)

# Plot the histogram of samples
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=30, density=True, alpha=0.7)
plt.plot(x, y, 'r-', lw=2)
plt.title('Normal Distribution: Random Samples')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)

# Statistical tests for normality
# Shapiro-Wilk test
stat, p_value = stats.shapiro(samples)
print(f"Shapiro-Wilk Test: statistic={stat:.4f}, p-value={p_value:.4f}")
# If p-value > 0.05, data is likely normal
```

## 2. Binomial Distribution

### Description
The binomial distribution models the number of successes in a fixed number of independent Bernoulli trials, each with the same probability of success.

### Use Cases
- Number of heads in a series of coin flips
- Number of defective items in quality control testing
- Number of successful sales calls
- Election results (voters selecting candidate A vs B)
- Patient recovery (success/failure) after a medical treatment

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Parameters
n = 20  # number of trials
p = 0.3  # probability of success

# Calculate PMF values
x = np.arange(0, n+1)
pmf = stats.binom.pmf(x, n, p)

# Plot PMF
plt.figure(figsize=(10, 6))
plt.bar(x, pmf, alpha=0.7)
plt.title(f'Binomial Distribution (n={n}, p={p})')
plt.xlabel('Number of Successes')
plt.ylabel('Probability')
plt.grid(True, alpha=0.3)

# Generate random samples
samples = np.random.binomial(n, p, 1000)

# Plot the histogram of samples
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=range(0, n+2), density=True, alpha=0.7)
plt.title(f'Binomial Distribution: Random Samples (n={n}, p={p})')
plt.xlabel('Number of Successes')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Mean and variance
mean = n * p
variance = n * p * (1 - p)
print(f"Theoretical Mean: {mean}")
print(f"Sample Mean: {np.mean(samples):.4f}")
print(f"Theoretical Variance: {variance}")
print(f"Sample Variance: {np.var(samples):.4f}")
```

## 3. Poisson Distribution

### Description
The Poisson distribution models the number of events occurring in a fixed interval of time or space, assuming these events occur with a known constant mean rate and independently of each other.

### Use Cases
- Number of calls received per hour at a call center
- Number of defects per unit area in material production
- Website traffic (visitors per minute)
- Number of accidents at an intersection per month
- Number of mutations in a DNA sequence

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Parameter
lambda_param = 3  # average rate of occurrence

# Calculate PMF values
x = np.arange(0, 15)
pmf = stats.poisson.pmf(x, lambda_param)

# Plot PMF
plt.figure(figsize=(10, 6))
plt.bar(x, pmf, alpha=0.7)
plt.title(f'Poisson Distribution (λ={lambda_param})')
plt.xlabel('Number of Events')
plt.ylabel('Probability')
plt.grid(True, alpha=0.3)

# Generate random samples
samples = np.random.poisson(lambda_param, 1000)

# Plot the histogram of samples
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=range(0, 16), density=True, alpha=0.7)
plt.bar(x, pmf, alpha=0.5, color='red')
plt.title(f'Poisson Distribution: Random Samples (λ={lambda_param})')
plt.xlabel('Number of Events')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.legend(['Theoretical PMF', 'Samples'])

# Mean and variance
# For Poisson, mean = variance = lambda
print(f"Theoretical Mean: {lambda_param}")
print(f"Sample Mean: {np.mean(samples):.4f}")
print(f"Theoretical Variance: {lambda_param}")
print(f"Sample Variance: {np.var(samples):.4f}")
```

## 4. Exponential Distribution

### Description
The exponential distribution models the time between events in a Poisson process, or the waiting time for the next event to occur.

### Use Cases
- Time between customer arrivals at a store
- Lifetime of electronic components
- Time until the next earthquake occurs
- Wait time between calls at a call center
- Service times in queuing theory

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Parameter
lambda_param = 0.5  # rate parameter
# Mean = 1/lambda, Variance = 1/lambda^2

# Calculate PDF values
x = np.linspace(0, 10, 1000)
pdf = stats.expon.pdf(x, scale=1/lambda_param)

# Plot PDF
plt.figure(figsize=(10, 6))
plt.plot(x, pdf, 'r-', lw=2)
plt.title(f'Exponential Distribution (λ={lambda_param})')
plt.xlabel('Time')
plt.ylabel('Probability Density')
plt.grid(True)

# Generate random samples
samples = np.random.exponential(scale=1/lambda_param, size=1000)

# Plot the histogram of samples
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=30, density=True, alpha=0.7)
plt.plot(x, pdf, 'r-', lw=2)
plt.title(f'Exponential Distribution: Random Samples (λ={lambda_param})')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.grid(True)
plt.xlim(0, 10)

# Mean and variance
mean = 1/lambda_param
variance = 1/(lambda_param**2)
print(f"Theoretical Mean: {mean}")
print(f"Sample Mean: {np.mean(samples):.4f}")
print(f"Theoretical Variance: {variance}")
print(f"Sample Variance: {np.var(samples):.4f}")
```

## 5. Uniform Distribution

### Description
The uniform distribution assigns equal probability to all values within a given range.

### Use Cases
- Random number generation
- Simulation of random events with equal probabilities
- Rounding errors in numerical calculations
- Modeling uncertainty when only bounds are known
- Generating random positions in a defined space

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Parameters
a = 2  # lower bound
b = 8  # upper bound

# Calculate PDF values
x = np.linspace(a-1, b+1, 1000)
pdf = stats.uniform.pdf(x, loc=a, scale=b-a)

# Plot PDF
plt.figure(figsize=(10, 6))
plt.plot(x, pdf, 'r-', lw=2)
plt.title(f'Uniform Distribution [a={a}, b={b}]')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.grid(True)

# Generate random samples
samples = np.random.uniform(a, b, 1000)

# Plot the histogram of samples
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=30, density=True, alpha=0.7)
plt.plot(x, pdf, 'r-', lw=2)
plt.title(f'Uniform Distribution: Random Samples [a={a}, b={b}]')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)

# Mean and variance
mean = (a + b) / 2
variance = ((b - a)**2) / 12
print(f"Theoretical Mean: {mean}")
print(f"Sample Mean: {np.mean(samples):.4f}")
print(f"Theoretical Variance: {variance}")
print(f"Sample Variance: {np.var(samples):.4f}")
```

## 6. Log-Normal Distribution

### Description
The log-normal distribution is a continuous probability distribution of a random variable whose logarithm is normally distributed.

### Use Cases
- Stock prices and financial returns
- Income distributions and wealth
- Size distributions (organization size, biological measurements)
- Failure times in reliability analysis
- Particle size distribution in materials science

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Parameters
mu = 0  # mean of log(X)
sigma = 0.5  # standard deviation of log(X)

# Calculate PDF values
x = np.linspace(0, 5, 1000)
pdf = stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu))

# Plot PDF
plt.figure(figsize=(10, 6))
plt.plot(x, pdf, 'r-', lw=2)
plt.title(f'Log-Normal Distribution (μ={mu}, σ={sigma})')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.grid(True)

# Generate random samples
samples = np.random.lognormal(mu, sigma, 1000)

# Plot the histogram of samples
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=30, density=True, alpha=0.7)
plt.plot(x, pdf, 'r-', lw=2)
plt.title(f'Log-Normal Distribution: Random Samples (μ={mu}, σ={sigma})')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.xlim(0, 5)

# Mean and variance
mean = np.exp(mu + sigma**2/2)
variance = (np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2)
print(f"Theoretical Mean: {mean:.4f}")
print(f"Sample Mean: {np.mean(samples):.4f}")
print(f"Theoretical Variance: {variance:.4f}")
print(f"Sample Variance: {np.var(samples):.4f}")
```

## 7. Student's t-Distribution

### Description
The t-distribution is a continuous probability distribution that arises when estimating the mean of a normally distributed population when the sample size is small and the population standard deviation is unknown.

### Use Cases
- Small sample hypothesis testing
- Confidence intervals with small samples
- Regression analysis
- Testing for differences between means
- Robust statistical methods when data has heavier tails than normal

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Parameter
df = 5  # degrees of freedom

# Calculate PDF values
x = np.linspace(-5, 5, 1000)
pdf_t = stats.t.pdf(x, df)
pdf_norm = stats.norm.pdf(x)  # Normal for comparison

# Plot PDF
plt.figure(figsize=(10, 6))
plt.plot(x, pdf_t, 'r-', lw=2, label=f't-distribution (df={df})')
plt.plot(x, pdf_norm, 'b--', lw=2, label='Normal distribution')
plt.title(f'Student\'s t-Distribution (df={df})')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.grid(True)
plt.legend()

# Generate random samples
samples = stats.t.rvs(df, size=1000)

# Plot the histogram of samples
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=30, density=True, alpha=0.7)
plt.plot(x, pdf_t, 'r-', lw=2)
plt.title(f't-Distribution: Random Samples (df={df})')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.xlim(-5, 5)

# Mean and variance
# For t-distribution with df>1, mean=0, variance=df/(df-2) for df>2
if df > 1:
    print(f"Theoretical Mean: 0")
    print(f"Sample Mean: {np.mean(samples):.4f}")
if df > 2:
    variance = df / (df - 2)
    print(f"Theoretical Variance: {variance:.4f}")
    print(f"Sample Variance: {np.var(samples):.4f}")
else:
    print("Variance is undefined for df <= 2")
```

## 8. Chi-Square Distribution

### Description
The chi-square distribution is a continuous probability distribution of the sum of squares of k independent standard normal random variables.

### Use Cases
- Goodness-of-fit tests
- Test for independence in contingency tables
- Confidence intervals for population variance
- Quality control in manufacturing
- Feature selection in machine learning

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Parameter
df = 5  # degrees of freedom

# Calculate PDF values
x = np.linspace(0, 20, 1000)
pdf = stats.chi2.pdf(x, df)

# Plot PDF
plt.figure(figsize=(10, 6))
plt.plot(x, pdf, 'r-', lw=2)
plt.title(f'Chi-Square Distribution (df={df})')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.grid(True)

# Generate random samples
samples = stats.chi2.rvs(df, size=1000)

# Plot the histogram of samples
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=30, density=True, alpha=0.7)
plt.plot(x, pdf, 'r-', lw=2)
plt.title(f'Chi-Square Distribution: Random Samples (df={df})')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.xlim(0, 20)

# Mean and variance
mean = df
variance = 2 * df
print(f"Theoretical Mean: {mean}")
print(f"Sample Mean: {np.mean(samples):.4f}")
print(f"Theoretical Variance: {variance}")
print(f"Sample Variance: {np.var(samples):.4f}")
```

## 9. Beta Distribution

### Description
The beta distribution is a family of continuous probability distributions defined on the interval [0, 1], parameterized by two positive shape parameters, α and β.

### Use Cases
- Modeling probabilities or proportions
- Bayesian statistics as prior for binomial distribution
- Project management (task completion times)
- A/B testing analysis
- Modeling random variables with finite bounds

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Parameters
alpha_values = [0.5, 1, 2, 5]
beta_values = [0.5, 1, 2, 5]

# Plot different beta distributions
plt.figure(figsize=(12, 8))

x = np.linspace(0, 1, 1000)

for alpha in alpha_values:
    for beta in beta_values:
        pdf = stats.beta.pdf(x, alpha, beta)
        plt.plot(x, pdf, lw=2, label=f'α={alpha}, β={beta}')

plt.title('Beta Distributions with Different Parameters')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)

# Example with specific parameters
alpha, beta = 2, 5

# Generate random samples
samples = stats.beta.rvs(alpha, beta, size=1000)

# Plot the histogram of samples with theoretical PDF
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=30, density=True, alpha=0.7)
plt.plot(x, stats.beta.pdf(x, alpha, beta), 'r-', lw=2)
plt.title(f'Beta Distribution: Random Samples (α={alpha}, β={beta})')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)

# Mean and variance
mean = alpha / (alpha + beta)
variance = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
print(f"Theoretical Mean: {mean:.4f}")
print(f"Sample Mean: {np.mean(samples):.4f}")
print(f"Theoretical Variance: {variance:.4f}")
print(f"Sample Variance: {np.var(samples):.4f}")
```

## 10. Gamma Distribution

### Description
The gamma distribution is a two-parameter family of continuous probability distributions that models positive continuous variables.

### Use Cases
- Modeling waiting times when the rate varies
- Rainfall amounts
- Insurance claim sizes
- Reliability and lifetime modeling
- Bayesian hierarchical models

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Parameters
shape = 2  # shape parameter (k)
scale = 2  # scale parameter (θ)
# Alternatively can be parameterized with rate=1/scale

# Calculate PDF values
x = np.linspace(0, 15, 1000)
pdf = stats.gamma.pdf(x, shape, scale=scale)

# Plot PDF
plt.figure(figsize=(10, 6))
plt.plot(x, pdf, 'r-', lw=2)
plt.title(f'Gamma Distribution (shape={shape}, scale={scale})')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.grid(True)

# Generate random samples
samples = stats.gamma.rvs(shape, scale=scale, size=1000)

# Plot the histogram of samples
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=30, density=True, alpha=0.7)
plt.plot(x, pdf, 'r-', lw=2)
plt.title(f'Gamma Distribution: Random Samples (shape={shape}, scale={scale})')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.xlim(0, 15)

# Mean and variance
mean = shape * scale
variance = shape * scale**2
print(f"Theoretical Mean: {mean}")
print(f"Sample Mean: {np.mean(samples):.4f}")
print(f"Theoretical Variance: {variance}")
print(f"Sample Variance: {np.var(samples):.4f}")
```

## Practical Data Science Application: Selecting the Right Distribution

Here's a practical example of fitting distributions to real-world data:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Generate some example data (replace with your actual data)
# Let's simulate insurance claim amounts which often follow a log-normal distribution
np.random.seed(42)
data = np.random.lognormal(mean=7, sigma=1.2, size=500)

# Plot histogram of data
plt.figure(figsize=(12, 8))
plt.hist(data, bins=30, density=True, alpha=0.7, label='Data')

# Fit several distributions
distributions = [
    stats.norm,
    stats.lognorm,
    stats.gamma,
    stats.expon,
    stats.weibull_min
]

colors = ['red', 'green', 'blue', 'orange', 'purple']
x = np.linspace(min(data), max(data), 1000)

# Test each distribution and plot the fitted PDF
for i, dist in enumerate(distributions):
    # Fit distribution to data
    params = dist.fit(data)
    
    # Calculate AIC
    log_likelihood = np.sum(dist.logpdf(data, *params))
    k = len(params)
    aic = 2 * k - 2 * log_likelihood
    
    # Plot PDF
    pdf = dist.pdf(x, *params)
    plt.plot(x, pdf, color=colors[i], lw=2, label=f'{dist.name} (AIC: {aic:.2f})')
    
    # Kolmogorov-Smirnov test
    ks_stat, p_value = stats.kstest(data, lambda x: dist.cdf(x, *params))
    print(f"{dist.name}:")
    print(f"  Parameters: {params}")
    print(f"  KS statistic: {ks_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  AIC: {aic:.4f}")
    if p_value > 0.05:
        print("  This distribution is a good fit (failed to reject)")
    else:
        print("  This distribution is not a good fit (rejected)")
    print()

plt.title('Distribution Fitting Comparison')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

# QQ plots for visual goodness-of-fit assessment
plt.figure(figsize=(15, 10))

for i, dist in enumerate(distributions):
    params = dist.fit(data)
    
    plt.subplot(2, 3, i+1)
    
    # Create QQ plot
    theoretical_quantiles = dist.ppf(np.linspace(0.01, 0.99, 100), *params)
    empirical_quantiles = np.quantile(data, np.linspace(0.01, 0.99, 100))
    
    plt.scatter(theoretical_quantiles, empirical_quantiles, alpha=0.7)
    plt.plot([min(theoretical_quantiles), max(theoretical_quantiles)], 
             [min(theoretical_quantiles), max(theoretical_quantiles)], 
             'r--', lw=2)
    
    plt.title(f'QQ Plot: {dist.name}')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Empirical Quantiles')
    plt.grid(True)

plt.tight_layout()
plt.show()
```

## Additional Considerations for Distribution Selection

When choosing a probability distribution for modeling data:

1. **Domain knowledge**: Consider the natural constraints on your data (e.g., positive values, bounded between 0 and 1)

2. **Data characteristics**:
   - Symmetry/skewness
   - Bounds (finite or infinite)
   - Discrete vs. continuous
   - Tail behavior (light or heavy tails)

3. **Statistical tests**:
   - Kolmogorov-Smirnov test
   - Anderson-Darling test
   - Chi-square goodness-of-fit test

4. **Information criteria**:
   - Akaike Information Criterion (AIC)
   - Bayesian Information Criterion (BIC)

5. **Visual methods**:
   - QQ plots
   - PP plots
   - Histogram with PDF overlay

6. **Real-world interpretability**: The chosen distribution should make sense for the phenomenon being modeled
