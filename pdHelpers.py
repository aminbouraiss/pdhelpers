"""
A collection of helper methods for pandas DataFrames.

"""

import pandas as pd
import re
import numpy as np
import sys
import os
from datetime import datetime, date
from IPython.core.display import display, HTML


class Helpers(object):
    """Pandas helper methods"""


    def round_floats(self):
        """adds a thousand separators to pandas output."""
        pd.set_option('display.float_format', self.Display_Thousands_Precision)

    def columnsWidth(self, width=150):
        """Sets the maximum width of the columns.

        :param width: The display width (defaults to 150)
        :type width: int

        :return: None

        """
        pd.set_option('max_colwidth', width)

    def maxColumns(self, maxCols=40):
        """Sets the maximum number of truncated columns (defaults to 40).

        :param maxCols: The maximum number of untruncated columns
        :type width: int

        :return: None

        """
        pd.set_option('display.max_columns', maxCols)


    def Display_Thousands_Precision(self, num, floatprecision=2):
        """Format a float number with thousand separator.

        :param num: a float number
        :type num: float        
        :param floatprecision: The float precision (default 2)
        :type floatprecision: Integer

        :return: formated string. ex: 10,000.00

        """
        formatStr = ":0,.{}f".format(floatprecision)
        formatStr = '{' + formatStr + '}'
        return formatStr.format(num)

    def __toHtml__(self,htmlCode):
        """print HTML to to jupyter notebook.

            Args:
                htmlCode: The html to render.

            Returns:
                outputs HTML.

        """
        display(HTML(htmlCode))

    def __getValues__(self,df):
        """Get the value count for each column"""
        for col in df.columns:
            countSerie = df[col].value_counts(dropna=False)
            countDf = (pd.DataFrame(countSerie)
                       .reset_index()
                       .rename(columns={'index': 'Value', col: 'Count'})
                       .assign(Column=col))
            yield countDf


    def showValueCount(self,df):
        """Displays the value count (including NaNs) for each column.

        :param df: The Pandas DataFrame to scan.
        :type df: Pandas DataFrame

        :return: None
        """
        dfList = list(self.__getValues__(df))
        for colDf in dfList:
            col = colDf.Column.unique()[0]
            title = "<h4 style='color:blue'>Column {0}</h4>".format(col)
            html = title + colDf.to_html()
            self.__toHtml__(html)


    def valueCount(self,df):
        """Returns the value count (including NaNs) for each column.

        :param df: The Pandas DataFrame to scan.
        :type df: Pandas DataFrame

        :return: A dict containing the value count for each column
        """
        dfDict = dict((dframe.Column.unique()[0], dframe)
                      for dframe in list(self.__getValues__(df)))
        return dfDict


    def searchCols(self, df, pattern):
        """Look for columns matching a regex pattern.

        :param df: A pandas DataFrame.
        :type df: Pandas DataFrame

        :param pattern: Regex pattern (case insensitive).
        :type pattern: String

        :return: A numpy array containing the matching columns
        :rtype: object        

        """
        selector = df.columns.str.contains(pattern, flags=re.I)
        selection = df.columns[selector]
        return selection


    def describeDf(self, df):
        """Returns the following values for each dataframe column:

                - Uniques
                - Max
                - Sum
                - Dtype

        :param df: A pandas DataFrame.
        :type df: Pandas DataFrame

            Args:
                df: The data frame to analyze.

        :return: Pandas DataFrame.
        :rtype: object        

        """
        metrics = self.findMetrics(df)
        dfDict = {}
        for col in df.columns:
            series = df[col]
            try:
                uniqueCount = series.value_counts(dropna=False)
                dfDict[col] = [uniqueCount]
                if str(series.dtype) == 'datetime64[ns]':
                    dfDict[col] = [dfDict[col][0], series.max()]
                else:
                    dfDict[col].append(series.max())
                if col in metrics:
                    seriesSum = float(series.sum())
                    dfDict[col] = [val for val in dfDict[col]] + [seriesSum]
                else:
                    dfDict[col] = [val for val in dfDict[col]] + ["N/A"]

                typeVal = series.dtype.name
                dfDict[col] = [val for val in dfDict[col]] + [typeVal]
            except:
                print(col)
                raise
        newDf = pd.DataFrame(dfDict, index=['Uniques', 'Max', 'Sum', 'Dtype'])
        return newDf

    def findDimensions(self, df):
        """Find a DataFrame's dimensions, columns matching one the following dtypes:

                - datetime64
                - category
                - object
                - datetime

        :param df: A pandas DataFrame.
        :type df: Pandas DataFrame

        :return: Matching column names
        :rtype: list

        """
        dimTypes = ['datetime64[ns]', 'category', 'object', 'datetime']
        dimTypes = ','.join(dimTypes)
        typecheck = df.dtypes
        dimBools = typecheck.apply(lambda x: str(x) in dimTypes)
        dimensions = df.columns[dimBools].tolist()
        return dimensions

    def findMetrics(self, df):
        """Find a DataFrame's metrics, columns matching not matching one of the following dtypes:

                - datetime64
                - category
                - object
                - datetime

        :param df: A pandas DataFrame.
        :type df: Pandas DataFrame

        :return: The matching column names.
        :rtype: list                

        """
        dims = set(self.findDimensions(df))
        columnsSet = set(df.columns)
        metrics = columnsSet.difference(dims)
        return list(metrics)

    def generate_Df(self):
        """Generate a sample pandas DataFrame with the following column types:

                - A           float64
                - B    datetime64[ns]
                - C           float32
                - D             int32
                - E          category
                - F            object

        :return: A pandas DataFrame
        :rtype: object     

        """
        df = pd.DataFrame({'A': 1., 'B': pd.Timestamp('20130102'),
                           'C': pd.Series(1, index=list(range(4)),
                                          dtype='float32'),
                           'D': np.array([3] * 4, dtype='int32'),
                           'E': pd.Categorical(["test", "train", "test", "train"]),
                           'F': 'foo'})
        return df

    def commonCols(self, df1, df2):
        """Perform a discrepancy check between two data frames
            It compares on the common dimensions and metrics
            between the two data frames

        :param df1: The first dataframe on which to perform the test.
        :type df: Pandas DataFrame            
        :param df2: The data frame to compare against.
        :type df2: Pandas DataFrame

        :return: a dict with three keys:


                - commonDims: The dimensions present in both columns.both.
                - commonMetrics: The metrics present in both columns.
                - allcolsCommon: The columns present in df1 not present in df2 (metrics + dimensions)

        :rtype: dict   

        """
        df1Dims = set((self.findDimensions(df1)))
        df1Metrics = set(df1.columns.difference(df1Dims))
        df2Dims = set(self.findDimensions(df2))
        df2Metrics = set(df2.columns.difference(df2Dims))
        commonDims = df2Dims.intersection(df1Dims)
        commonMetrics = df2Metrics.intersection(df1Metrics)
        allcolsCommon = df1.columns.intersection(df2.columns)

        returnDict = dict(commonDims=list(commonDims),
                          commonMetrics=list(commonMetrics),
                          commonCols=list(allcolsCommon))
        return returnDict

    def diffCols(self, df1, df2):
        """Get the names of the columns present in the first dataframe specified and absent in the second.

        :param df1: The first dataframe on which to perform the test.
        :type df: Pandas DataFrame            
        :param df2: The data frame to compare against.
        :type df2: Pandas DataFrame

        :return: A dict with three keys:


                * dimDiff: The dimensions present in df1 not present in df2
                * metricDiff: The metrics present in df1 not present in df2
                * allcolsDif: The columns present in df1 not present in df2 (metrics + dimensions)


        :rtype: dict   

        """

        df1Dims = set((self.findDimensions(df1)))
        df1Metrics = set(df1.columns.difference(df1Dims))
        df2Dims = set(self.findDimensions(df2))
        df2Metrics = set(df2.columns.difference(df2Dims))
        dimDiff = df1Dims.difference(df2Dims)
        metricDiff = df1Metrics.difference(df2Metrics)
        allcolsDif = df1.columns.difference(df2.columns)

        returnDict = dict(dimDiff=list(dimDiff),
                          metricDiff=list(metricDiff), allcolsDif=list(allcolsDif))
        return returnDict

    def deleteCol(self, df, col):
        """Safely delete a DataFrame column whithout raising an error
        if the column doesn't exist

        :param df: Pandas DataFrame.
        :type df: Pandas DataFrame  
        :param col: The name of the column to delete.
        :type df: string          

        """
        column = col in df.columns
        if column is not False:
            del df[col]

    def appendNewDates(self, historicDf, newDf, index_column=None):
        """Append new rows  containing newer dates to a historical DataFrame.

        :param historicDf: The DataFrame containing the historical data.
        :type historicDf: Pandas DataFrame  
        :param newDf: The DF containing the new rows to append.
        :type newDf: Pandas DataFrame  
        :param index_column: Append by the specified column name if specified, otherwise by the DataFrame's index.
        :type index_column: string

        :return: A pandas DataFrame
        :rtype: object             

        """
        historic = historicDf.copy()
        if index_column:            
            appenDedDf = pd.concat([historic[~historic[index_column].isin(newDf[index_column])], newDf])
        else:
            appenDedDf = pd.concat([historic[~historic.index.isin(newDf.index)], newDf])
        return appenDedDf


    def setDateIndex(self, df, targetCol):
        """Converts a date column to a datetime format and sets it as a sorted index.

        :param df: The data frame to transform.
        :type df: Pandas DataFrame
        :param targetCol: The column to convert to the datetime format.
        :type targetCol: string        

        :return: A pandas DataFrame
        :rtype: object 

            """
        formatted = df.copy()
        formatted['Date'] = pd.to_datetime(formatted[targetCol])
        formatted.set_index('Date', inplace=True)
        formatted.sort_index(ascending=True, inplace=True)
        del formatted[targetCol]
        return formatted


if __name__ == '__main__':
    utilities = Helpers()