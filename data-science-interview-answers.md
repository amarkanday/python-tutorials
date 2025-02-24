# Data Science Interview Questions and Answers

## Statistics and Probability

### Bias vs. Variance
**Bias** represents systematic errors in a model's predictions - when a model consistently misses its target. High bias models are typically too simple (underfitting). **Variance** represents sensitivity to fluctuations in the training data - how much predictions change with different training samples. High variance models are typically too complex (overfitting). The goal is to find the optimal balance between the two.

### P-value
A p-value is the probability of obtaining results at least as extreme as those observed, assuming the null hypothesis is true. In hypothesis testing, if the p-value is below a pre-defined significance level (typically 0.05), we reject the null hypothesis. It quantifies the statistical significance of our findings but doesn't measure the size of an effect or its practical importance.

### Central Limit Theorem
The Central Limit Theorem states that the sampling distribution of the mean of a sufficiently large number of independent random variables will approximate a normal distribution, regardless of the original distribution. This holds true provided the original distribution has finite variance and the sample size is large enough (typically n > 30).

### Types of Distributions
- **Normal/Gaussian**: Symmetric, bell-shaped; used for natural phenomena
- **Binomial**: Discrete, counts successes in fixed trials; used for binary outcomes
- **Poisson**: Discrete, models rare events; used for count data in fixed time/space
- **Exponential**: Continuous, models time between events; used for waiting times
- **Uniform**: Equal probability across range; used for random number generation
- **Log-normal**: Skewed right; used for multiplicative processes like stock prices

### Handling Imbalanced Datasets
- **Resampling**: Undersample majority class or oversample minority class
- **Synthetic data generation**: SMOTE or ADASYN to create synthetic minority examples
- **Algorithm-level approaches**: Use algorithms less sensitive to imbalance
- **Cost-sensitive learning**: Assign higher misclassification costs to minority class
- **Ensemble methods**: Combine multiple models optimized for different aspects
- **Evaluation metrics**: Use F1-score, precision-recall AUC instead of accuracy

### Type 1 vs. Type 2 Errors
**Type 1 error** (false positive): Rejecting a true null hypothesis (finding an effect that doesn't exist).
**Type 2 error** (false negative): Failing to reject a false null hypothesis (missing an effect that does exist).
The tradeoff between these errors is controlled by the significance level (α).

### Bayesian vs. Frequentist Statistics
**Frequentist** statistics interprets probability as long-run frequency of events and relies on p-values and confidence intervals. It treats parameters as fixed but unknown constants.
**Bayesian** statistics interprets probability as a degree of belief and uses prior distributions updated with evidence to generate posterior distributions. It treats parameters as random variables with distributions.

## Machine Learning

### Linear Regression
Linear regression models the relationship between a dependent variable and one or more independent variables by fitting a linear equation: Y = β₀ + β₁X₁ + β₂X₂ + ... + ε. The coefficients (β) are estimated using least squares, minimizing the sum of squared differences between observed and predicted values.

### Supervised vs. Unsupervised Learning
In **supervised learning**, the algorithm learns from labeled training data to make predictions or decisions (classification, regression). Examples: linear regression, decision trees, neural networks.
In **unsupervised learning**, the algorithm finds patterns or structures in unlabeled data. Examples: clustering, dimensionality reduction, association rules.

### Decision Tree Algorithm
Decision trees partition the feature space into regions by making sequential binary splits that maximize information gain (or minimize impurity). At each node, the algorithm selects the feature and threshold that best separates the data. Predictions are made by navigating from the root to a leaf node. They're interpretable but prone to overfitting.

### Bias-Variance Tradeoff
The bias-variance tradeoff involves finding the model complexity that minimizes total error. Simple models have high bias (underfitting) but low variance. Complex models have low bias but high variance (overfitting). Total error = bias² + variance + irreducible error. The goal is to find the sweet spot that minimizes this sum.

### Preventing Overfitting
- **Cross-validation**: Train on multiple data subsets
- **Regularization**: Add penalty terms (L1, L2) to the loss function
- **Pruning**: Remove unnecessary complexity from models (especially trees)
- **Early stopping**: Halt training when validation performance degrades
- **Ensemble methods**: Combine multiple models to reduce variance
- **Dropout**: Randomly disable neurons during neural network training
- **Feature selection**: Reduce dimensionality by using only relevant features

### Random Forest
Random Forest is an ensemble method that builds multiple decision trees and merges their predictions. It introduces randomness by:
1. Bootstrap aggregating (bagging): Training each tree on a random subset of data
2. Feature randomness: Considering only a random subset of features at each split

This reduces overfitting, handles high-dimensional data well, and provides feature importance metrics.

### Gradient Descent
Gradient descent is an optimization algorithm that iteratively adjusts model parameters to minimize a loss function. It calculates the gradient (direction of steepest increase) of the loss function with respect to each parameter, then updates parameters in the opposite direction of the gradient. The learning rate controls the step size. Variants include batch, stochastic, and mini-batch gradient descent.

### Neural Network
Neural networks consist of interconnected layers of neurons that transform input data through weighted connections and activation functions. The network learns by:
1. Forward propagation: Passing inputs through layers to generate predictions
2. Comparing predictions to actual values using a loss function
3. Backward propagation: Computing gradients and updating weights to minimize loss

Deep networks with multiple hidden layers can learn hierarchical representations of complex data.

### Clustering Algorithms
- **K-means**: Partitions data into k clusters by minimizing within-cluster variance
- **Hierarchical clustering**: Builds nested clusters by merging or splitting
- **DBSCAN**: Density-based clustering that handles irregular shapes and noise
- **Gaussian Mixture Models**: Probabilistic model assuming data comes from multiple Gaussian distributions
- **Spectral clustering**: Uses eigenvalues of similarity matrices for dimensionality reduction before clustering

### Choosing ML Algorithms
I consider:
1. Problem type (classification, regression, clustering)
2. Data characteristics (size, dimensionality, sparsity)
3. Interpretability requirements
4. Computational constraints
5. Performance metrics that matter
6. Data assumptions (linearity, independence)
7. Handling of missing values and categorical features

Then I typically test multiple algorithms using cross-validation to find the best performer.

### Regularization
Regularization adds constraints to a model to prevent overfitting by penalizing complexity. Common methods include:
- **L1 (Lasso)**: Adds the absolute sum of coefficients to the loss function, encouraging sparsity
- **L2 (Ridge)**: Adds the squared sum of coefficients, shrinking all parameters toward zero
- **Elastic Net**: Combines L1 and L2 penalties
- **Dropout**: Randomly deactivates neurons during training
- **Early stopping**: Halts training when validation performance plateaus

## Programming and SQL

### Python Function for Data Preprocessing

```python
def preprocess_data(df):
    """
    Clean and preprocess a pandas dataframe
    
    Parameters:
    df (pandas.DataFrame): Input dataframe
    
    Returns:
    pandas.DataFrame: Cleaned and preprocessed dataframe
    """
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Handle missing values
    numerical_cols = df_clean.select_dtypes(include=['int', 'float']).columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    
    # Replace numerical missing values with median
    for col in numerical_cols:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    # Replace categorical missing values with mode
    for col in categorical_cols:
        df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    
    # Convert categorical variables to dummy variables
    df_clean = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
    
    # Remove outliers using IQR method for numerical columns
    for col in numerical_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    # Scale numerical features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_clean[numerical_cols] = scaler.fit_transform(df_clean[numerical_cols])
    
    return df_clean
```

### SQL Query Example

```sql
SELECT 
    c.customer_name,
    COUNT(o.order_id) AS total_orders,
    SUM(o.order_amount) AS total_spent,
    AVG(o.order_amount) AS avg_order_value,
    MAX(o.order_date) AS last_order_date
FROM 
    customers c
LEFT JOIN 
    orders o ON c.customer_id = o.customer_id
WHERE 
    o.order_date BETWEEN '2023-01-01' AND '2023-12-31'
    AND o.order_status = 'completed'
GROUP BY 
    c.customer_id, c.customer_name
HAVING 
    COUNT(o.order_id) > 3
ORDER BY 
    total_spent DESC
LIMIT 100;
```

### Handling Missing Values
- **Delete**: Remove rows or columns with significant missing data
- **Imputation**: Replace with mean/median/mode for numerical, mode for categorical
- **Prediction**: Use ML models to predict missing values
- **Advanced methods**: MICE (Multiple Imputation by Chained Equations)
- **Domain-specific**: Use domain knowledge to fill meaningful defaults
- **Flag missing values**: Add indicator columns showing which values were missing

### Optimizing SQL Queries
- **Indexing**: Create appropriate indexes on frequently queried columns
- **Query rewriting**: Avoid SELECT *, use specific columns
- **JOIN optimization**: Ensure proper join conditions and order
- **Subqueries**: Replace with JOINs where appropriate
- **LIMIT clause**: Add early when only needing a sample
- **Avoid functions on indexed columns**: They prevent index usage
- **EXPLAIN/ANALYZE**: Use to identify bottlenecks
- **Materialized views**: Pre-compute frequent complex queries
- **Partitioning**: Split large tables into manageable chunks

### SQL Join Types
- **INNER JOIN**: Returns only matching rows from both tables
- **LEFT JOIN**: Returns all rows from left table and matching rows from right table
- **RIGHT JOIN**: Returns all rows from right table and matching rows from left table
- **FULL OUTER JOIN**: Returns all rows from both tables, regardless of matches
- **CROSS JOIN**: Returns Cartesian product of both tables (all possible combinations)

### Python Factorial Function

```python
def factorial(n):
    """
    Compute the factorial of a non-negative integer n
    
    Parameters:
    n (int): Non-negative integer
    
    Returns:
    int: n factorial (n!)
    
    Raises:
    ValueError: If n is negative
    """
    if not isinstance(n, int):
        raise TypeError("Input must be an integer")
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n-1)

# Alternative iterative implementation
def factorial_iterative(n):
    if not isinstance(n, int):
        raise TypeError("Input must be an integer")
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
```

## Data Analysis and Problem Solving

### Exploratory Data Analysis Approach
1. **Initial data understanding**: Size, shape, types, basic statistics
2. **Data quality assessment**: Missing values, duplicates, outliers
3. **Univariate analysis**: Distribution of individual variables (histograms, box plots)
4. **Bivariate analysis**: Relationships between variables (scatter plots, correlation)
5. **Feature engineering**: Create meaningful derived variables
6. **Pattern identification**: Look for trends, seasonality, clusters
7. **Hypothesis generation**: Develop questions to test based on observations
8. **Visualization**: Create informative plots to communicate findings

### Designing an A/B Test
1. **Define metrics**: Primary and secondary success metrics
2. **Formulate hypotheses**: Null and alternative, expected effect
3. **Calculate sample size**: Based on minimum detectable effect, power, significance
4. **Randomization strategy**: User-level vs. session-level assignment
5. **Experiment duration**: Based on user cycles and statistical power
6. **Implementation**: Split traffic, gather data, monitor for issues
7. **Analysis**: Hypothesis testing, segmentation analysis, check assumptions
8. **Documentation**: Record methodology, results, and recommendations

### Handling Outliers
1. **Identify outliers**: Statistical methods (Z-score, IQR) or visualization
2. **Investigate root causes**: Data entry errors, measurement issues, or valid extremes
3. **Decision-making**: Based on outlier type and analysis goals
   - Remove if erroneous or highly influential
   - Transform data (log, sqrt) to reduce impact
   - Cap at percentiles (winsorizing)
   - Use robust statistical methods
   - Create separate models for outlier groups if meaningful

### Communicating Findings to Non-Technical Audience
1. **Focus on business impact**: Lead with key insights and recommendations
2. **Use storytelling**: Create a narrative around the data
3. **Visual simplicity**: Clean, labeled charts without technical jargon
4. **Avoid statistical complexity**: Translate p-values and coefficients into plain language
5. **Use analogies**: Relate complex concepts to familiar scenarios
6. **Interactive demonstrations**: Allow exploration of key relationships
7. **Prepare for questions**: Anticipate concerns and have supporting details ready

### Data-Driven Problem Solving
For a business problem like "How can we reduce customer churn?":

1. **Problem definition**: Clearly define churn and its business impact
2. **Data collection**: Gather customer behavior, demographics, support interactions
3. **Exploratory analysis**: Identify patterns in users who churn vs. retain
4. **Feature engineering**: Create meaningful metrics like usage frequency, support tickets
5. **Model building**: Develop predictive models for churn probability
6. **Evaluation**: Test models using appropriate metrics (AUC, precision-recall)
7. **Insights generation**: Identify key drivers of churn
8. **Action plan**: Develop interventions for high-risk customers
9. **Implementation**: Deploy model in production with monitoring
10. **Feedback loop**: Measure intervention effectiveness and refine

### Handling High Cardinality Categorical Variables
1. **Frequency-based encoding**: Replace categories with their frequency
2. **Target encoding**: Replace with target mean for that category
3. **Clustering similar categories**: Group by semantic similarity
4. **Hierarchical grouping**: Create meaningful super-categories
5. **Embedding techniques**: Learn low-dimensional representations
6. **Feature hashing**: Map categories to fixed-size vector
7. **Decision tree-based methods**: Use algorithms less affected by cardinality (Random Forest)
8. **Regularization**: Apply stronger regularization to one-hot encoded variables
