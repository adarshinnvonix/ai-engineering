import pandas as pd
import numpy as np

# function to get simple series from an array
def getSeries(arr: np.array):
    print(pd.Series(arr))

# function to get series from an array with custom index
def getCustomSeries(arr: np.array, index: list):
    print(pd.Series(arr, index = index))    

# function to get series from a scalar value
def getScalarSeries(scalar: int):
    print(pd.Series(scalar, index = [0, 1, 2, 3, 4]))

# function to get series from a dictionary
def getSeriesFromDict(data: dict):
    print(pd.Series(data))

# function to create dataframe from a dictionary
def createDataFrame(data: dict):
    df = pd.DataFrame(data)
    print(df)