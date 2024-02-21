import pandas as pd
from abc import ABC, abstractmethod


class Imputer(ABC):
    """
    Base imputer class similar to the `scikit-learn` library imputers
    """    
    @abstractmethod
    def fit(self, input: pd.DataFrame) -> None:
        pass
    
    @abstractmethod
    def transform(self, input: pd.DataFrame) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def fit_transform(self, input: pd.DataFrame) -> pd.DataFrame:
        pass


class DailyMedianImputer(Imputer):
    """
    Uses the median value for per day to fill missing values.
    """    
    
    def __init__(self, fill_vals: pd.DataFrame=None) -> None:
        """Initialize the imputer, optionally with preset imputation values.

        Parameters
        ----------
        fill_vals : pd.DataFrame, optional
            Set the values to use for imputation.
        """        
        self.fill_vals: pd.DataFrame = fill_vals
    
    def fit(self, input: pd.DataFrame) -> None:
        """Fits the imputation values to the input. This overwrites the previous values.

        Parameters
        ----------
        input : pd.DataFrame
            Dataframe to fit. Must have a column/index named `day`.
        """        
        self.fill_vals = input.groupby('day').median()
        self.fill_vals = self.fill_vals.fillna(method='ffill', axis=1) # Ensure there are no NaN values left
        
    def transform(self, input: pd.DataFrame) -> pd.DataFrame:
        """Impute the missing values according to the fitted imputation values.

        Parameters
        ----------
        input : pd.DataFrame
            Dataframe to transform

        Returns
        -------
        pd.DataFrame
            Dataframe with values imputed
        """
        assert self.fill_vals is not None, "Imputer must be fitted before transformation"        
        output = input.fillna(self.fill_vals)
        return output.fillna(method='ffill', axis=0)
    
    def fit_transform(self, input: pd.DataFrame) -> pd.DataFrame:
        """ First fit the model to the input and then transform it.

        Parameters
        ----------
        input : pd.DataFrame
            Dataframe to fit and transform

        Returns
        -------
        pd.DataFrame
            Dataframe with values imputed
        """        
        self.fit(input)
        return self.transform(input)
    
    
    
class MedianImputer(Imputer):
    def __init__(self, fill_vals: pd.DataFrame = None) -> None:
        self.fill_vals: pd.DataFrame = fill_vals
    
    def fit(self, input: pd.DataFrame) -> None:
        self.fill_vals = input.median()
    
    def transform(self, input: pd.DataFrame) -> pd.DataFrame:
        return input.fillna(self.fill_vals)

    def fit_transform(self, input: pd.DataFrame) -> pd.DataFrame:
        self.fit(input)
        return self.transform(input)
