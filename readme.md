# Pandas Helpers

Pandas Helper is a collection of helper methods for pandas DataFrames.

## Requirements:

 - iPython https://ipython.org
 - numpy http://www.numpy.org/
 - pandas https://pandas.pydata.org

## Install

### Simple installation

To install this package run the following command:
```
$ pip install git+https://github.com/aminbouraiss/pdhelpers.git
```

### From a cloned repository

To install it from a cloned repository, run this command:

```
$ git clone https://github.com/aminbouraiss/pdhelpers.git
```

## Notebook example :

An [IPython (Jupyter)](http://ipython.org/) notebook showing this package usage is available at:

 - http://nbviewer.ipython.org/github/aminbouraiss/pdhelpers/blob/master/pdhelpers_examples.ipynb

## Script examples :
### Get common columns between two dataframes

Perform a discrepancy check between two data frames, it compares on the common dimensions and metrics 
between the two data frames.

This method returns a dict with three keys:

* **commonDims**: The dimensions present in both columns.both.
* **commonMetrics**: The metrics present in both columns.
* **commonCols**: The columns present in df1 not present in df2 (metrics + dimensions)

### Get diverging columns between two dataframes

Get the names of the columns present in the first dataframe specified and absent in the second.

This method returns a dict with three keys:

* **dimDiff**: The dimensions present in df1 not present in df2.
* **metricDiff**: The metrics present in df1 not present in df2.
* **allcolsDif**: The columns present in df1 not present in df2 (metrics + dimensions).


```python
import pdHelpers
import pprint

# Instantiate the pandas helper module
helpers = pdHelpers.Helpers()

# Generate the dataFrames
df1 = helpers.generate_Df()
df2 = (df[['B','D']]
         .assign(H=df.D*3))

# Print the two dataframes
separation = "_" * 40
print("df1\n{0}\n\n{1}\n".format(separation,df1))
print("df2\n{0}\n\n{1}\n".format(separation,df2))

# find the commmon columns
common = helpers.commonCols(df1,df2)
print("Common columns\n{0}\n".format(separation))
pprint.pprint(common)

# find the diverging columns
difference = helpers.diffCols(df1,df2)
print("\nDiverging columns\n{0}\n".format(separation))
pprint.pprint(difference)
```

    df1
    ________________________________________
    
         A          B    C  D      E    F
    0  1.0 2013-01-02  1.0  3   test  foo
    1  1.0 2013-01-02  1.0  3  train  foo
    2  1.0 2013-01-02  1.0  3   test  foo
    3  1.0 2013-01-02  1.0  3  train  foo
    
    df2
    ________________________________________
    
               B  D  H
    0 2013-01-02  3  9
    1 2013-01-02  3  9
    2 2013-01-02  3  9
    3 2013-01-02  3  9
    
    Common columns
    ________________________________________
    
    {'commonCols': ['B', 'D'], 'commonDims': ['B'], 'commonMetrics': ['D']}
    
    Diverging columns
    ________________________________________
    
    {'allcolsDif': ['A', 'C', 'E', 'F'],
     'dimDiff': ['E', 'F'],
     'metricDiff': ['A', 'C']}


## Generate a sample dataFrame

Generate a sample pandas DataFrame with the following column types:


* float64
* datetime64[ns]
* float32
* int32
* int32
* category
* object 


```python
import pdHelpers

# Instantiate the pandas helper module
helpers = pdHelpers.Helpers()

# Generate a sample dataFrame
df = helpers.generate_Df()

df.info()

df
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 4 entries, 0 to 3
    Data columns (total 6 columns):
    A    4 non-null float64
    B    4 non-null datetime64[ns]
    C    4 non-null float32
    D    4 non-null int32
    E    4 non-null category
    F    4 non-null object
    dtypes: category(1), datetime64[ns](1), float32(1), float64(1), int32(1), object(1)
    memory usage: 260.0+ bytes
### Round floats

This method automatically rounds values displayed by pandas 


```python
import random 
import pdHelpers
helpers = pdHelpers.Helpers()


df = helpers.generate_Df() # Generate a test DataFrame
df.C = df.C.apply(lambda x: random.uniform(1, 10))
df.C
```




    0    3.400638
    1    7.984441
    2    7.023958
    3    7.388083
    Name: C, dtype: float64




```python
helpers.round_floats() # Automatically round floats
df.C 
```




    0   3.40
    1   7.98
    2   7.02
    3   7.39
    Name: C, dtype: float64




```python
df.C.tolist() # The exported data is untouched
```




    [3.400637971393313, 7.98444055902644, 7.023957929542492, 7.38808312498648]
### Convert a column to datetime index

Converts a date column to a datetime format and sets it as a sorted index.


```python
import numpy as np
import pandas as pd
import pdHelpers

# Instantiate the pandas helper module
helpers = pdHelpers.Helpers()

# Create the time series
ts = ('2012-04-0{}'.format(x) for x in range(1,9))

# Generate the dataFrame
df = pd.DataFrame(dict(Dates=list(ts), Values=[
                        np.random.randint(1, 10) for x in range(1,9)]))

# convert the date column and set it as index
indexed_df = helpers.setDateIndex(df,'Dates')

# Print the results
separation = "_" * 40
print("The original dataframe dtypes\n{0}".format(separation))
info = df.info()
print("\nThe original dataFrame\n{0}\n\n{1}\n".format(separation,df))
print("\nThe converted dataFrame\n{0}\n\n{1}\n".format(separation,indexed_df))

```

    The original dataframe dtypes
    ________________________________________
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8 entries, 0 to 7
    Data columns (total 2 columns):
    Dates     8 non-null object
    Values    8 non-null int64
    dtypes: int64(1), object(1)
    memory usage: 200.0+ bytes
    
    The original dataFrame
    ________________________________________
    
            Dates  Values
    0  2012-04-01       2
    1  2012-04-02       1
    2  2012-04-03       3
    3  2012-04-04       3
    4  2012-04-05       9
    5  2012-04-06       6
    6  2012-04-07       3
    7  2012-04-08       1
    
    
    The converted dataFrame
    ________________________________________
    
                Values
    Date              
    2012-04-01       2
    2012-04-02       1
    2012-04-03       3
    2012-04-04       3
    2012-04-05       9
    2012-04-06       6
    2012-04-07       3
    2012-04-08       1    
    
    
