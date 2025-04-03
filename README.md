# Python Data Science Tutorials

This repository contains a collection of Python tutorials and exercises focused on data science, machine learning, and data analysis. The tutorials are designed to help you learn and practice various data science concepts through hands-on examples. Tests are modeled on what one would expect in a ML coding interview.

## Project Structure

```
python-tutorials/
├── Tutorials/
│   ├── exercises/          # Jupyter notebooks with exercises
│   │   ├── test1.ipynb    # Data Analysis and Visualization
│   │   ├── test2.ipynb    # Machine Learning Basics
│   │   ├── test3.ipynb    # Advanced ML Algorithms
│   │   ├── test4.ipynb    # Time Series Analysis
│   │   ├── test5.ipynb    # Deep Learning
│   │   ├── test6.ipynb    # Natural Language Processing
│   │   ├── test7.ipynb    # Statistical Analysis
│   │   ├── test8.ipynb    # Data Science Case Studies
│   ├── core_tutorials/    # Fundamental tutorials for beginners
│   ├── source/            # Source code and utility functions
│   │   ├── data_utils.py  # Data loading and preprocessing utilities
│   │   ├── ml_utils.py    # Machine learning helper functions
│   │   └── viz_utils.py   # Visualization utilities
│   └── solutions/         # Solution notebooks for exercises
├── data/                  # Data files used in tutorials
├── docs/                  # Documentation files
│   ├── tutorials/        # Detailed tutorial documentation
├── requirements.txt       # Python package dependencies
└── README.md             # This file
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/python-tutorials.git
   cd python-tutorials
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Tutorials

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Navigate to the `Tutorials/exercises` directory to find the tutorial notebooks.

## Tutorials Folder Structure

The `Tutorials/` directory is organized into 3 main subdirectories:

### 1. Core Tutorials (`Tutorials/core_tutorials/`)

This directory contains fundamental tutorials designed for beginners and those who need to refresh their knowledge. These tutorials provide a solid foundation before moving on to the more advanced exercises.


### 2. Exercises (`Tutorials/exercises/`)

This directory contains Jupyter notebooks with hands-on exercises and practice problems. Each notebook is designed to be self-contained and includes:

- **Theory and Concepts**: Explanations of key concepts and algorithms
- **Code Examples**: Practical implementations with detailed comments
- **Practice Problems**: Exercises to reinforce learning
- **Data Loading**: Instructions for loading and preprocessing data
- **Visualization**: Examples of data visualization techniques
- **Model Building**: Step-by-step guides for building ML models
- **Evaluation**: Methods for evaluating model performance

The exercises progress from basic to advanced topics, allowing you to build your skills gradually.


### 3. Solutions (`Tutorials/solutions/`)

This directory contains completed solution notebooks for the exercises. The solutions include:

- **Complete Implementations**: Fully working code for all exercises
- **Alternative Approaches**: Different ways to solve the same problem
- **Best Practices**: Examples of efficient and clean code
- **Performance Optimizations**: Techniques for improving code efficiency
- **Explanations**: Detailed explanations of the solution approach

The solutions are provided as a reference after you've attempted the exercises yourself.

## Tutorials Overview

### 1. Data Analysis and Visualization (test1.ipynb)
- Data loading and preprocessing with Pandas
- Exploratory Data Analysis (EDA)
- Data visualization with Matplotlib and Seaborn
- Statistical analysis and hypothesis testing
- Hands-on exercises with real-world datasets

### 2. Machine Learning Basics (test2.ipynb)
- Introduction to scikit-learn
- Supervised learning algorithms
- Model evaluation and validation
- Feature engineering and selection
- Cross-validation techniques

### 3. Advanced ML Algorithms (test3.ipynb)
- Ensemble methods (Random Forest, XGBoost)
- Support Vector Machines
- Neural Networks basics
- Model optimization and tuning
- Advanced feature engineering

### 4. Time Series Analysis (test4.ipynb)
- Time series data preprocessing
- ARIMA and SARIMA models
- Seasonal decomposition
- Forecasting techniques
- Real-world time series applications

### 5. Deep Learning (test5.ipynb)
- Neural Networks fundamentals
- CNN for image processing
- RNN for sequential data
- Transfer learning
- Deep learning best practices

### 6. Natural Language Processing (test6.ipynb)
- Text preprocessing
- Word embeddings
- Text classification
- Sentiment analysis
- Named Entity Recognition

### 7. Statistical Analysis (test7.ipynb)
- Advanced statistical methods
- Hypothesis testing
- ANOVA and regression analysis
- Statistical modeling
- Experimental design

### 8. Data Science Case Studies (test8.ipynb)
- Real-world data science projects
- End-to-end data analysis
- Business case studies
- Model deployment
- Performance optimization

### 9. ML System Design (test9.ipynb)
- ML system architecture
- Model serving and deployment
- Scalability considerations
- Monitoring and maintenance
- Best practices for production

### 10. Advanced Topics (test10.ipynb)
- Advanced algorithms
- Optimization techniques
- Specialized applications
- Industry-specific solutions
- Emerging trends in data science


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all contributors who have helped improve these tutorials
- Special thanks to the open-source community for their valuable tools and libraries

## Contact

For questions or suggestions, please open an issue in the GitHub repository.
