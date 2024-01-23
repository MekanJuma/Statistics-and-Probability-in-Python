import numpy as np
# from scipy import stats
# import statistics
import math


class Statistics:
    def factorial(self, number):
        if number < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        result = 1
        for i in range(2, number + 1):
            result *= i
        return result
    
    def mean(self, collection):
        n = len(collection)
        if n == 0:
            raise ValueError("Collection is empty")
        return sum(collection) / n
    
    def median(self, collection):
        n = len(collection)
        if n == 0:
            raise ValueError("Collection is empty")
        sorted_collection = sorted(collection)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_collection[mid-1] + sorted_collection[mid]) / 2
        else:
            return sorted_collection[mid]
    
    def mode(self, collection, multimode=False):
        n = len(collection)
        if n == 0:
            raise ValueError("Collection is empty")
        counts = {}
        for x in collection:
            if x in counts:
                counts[x] += 1
            else:
                counts[x] = 1
        max_count = max(counts.values())
        modes = [x for x in counts if counts[x] == max_count]
        if not multimode and len(modes) == 1:
            return modes[0]
        else:
            return modes
    
    def std_variance(self, collection, is_sample=False):
        """Calculates the standard deviation and variance of a collection of data.

        Args:
            collection: A list or tuple of numerical data.
            is_sample: A boolean indicating whether the data represents a sample or population.

        Returns:
            A tuple containing the variance and standard deviation of the data.
        """
        n = len(collection)
        if n == 0:
            raise ValueError("Collection is empty")
        mean = sum(collection) / n
        if is_sample:
            divisor = n - 1
        else:
            divisor = n
        variance = sum((x - mean) ** 2 for x in collection) / divisor
        std = variance ** 0.5
        return (variance, std)

class Probability(Statistics):
    def percentile(self, collection):
        n = len(collection)
        if n == 0:
            raise ValueError("Data is empty")
        data = sorted(collection)
        q1 = self.median(data[:n//2])
        if n % 2 == 0:
            q3 = self.median(data[n//2:])
        else:
            q3 = self.median(data[n//2+1:])
        return (q1, q3)
    
    def iqr(self, q3, q1):
        return q3 - q1
    
    def binomial(self, n, p, k):
        """
        Returns the probability of getting k successes in n independent trials,
        where each trial has a probability of success p.
        """
        if k < 0 or k > n:
            raise ValueError("k must be between 0 and n")
        return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

prob = Probability()
data = [10, 20, 30, 40, 50]

q1, q3 = prob.percentile(data)
iqr = prob.iqr(q3, q1)
print(iqr, q1, q3)


#! Statistics
# Descriptive statistics
# Measures of central tendency (mean, median, mode)
# Measures of dispersion (variance, standard deviation, range, interquartile range)
# Hypothesis testing
# Confidence intervals
# Correlation and regression analysis
# Time series analysis
# Nonparametric statistics
# Experimental design and analysis
# Sampling techniques and sampling distributions
# ANOVA (Analysis of Variance)
# Quality control and process control
# Decision theory and Bayesian analysis
# Factor analysis and cluster analysis
# Multivariate analysis


