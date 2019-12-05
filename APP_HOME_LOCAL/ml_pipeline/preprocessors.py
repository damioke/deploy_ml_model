import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# categorical variables with NA in train set
CATEGORICAL_VARS_WITH_NA = [
    'MasVnrType', 'BsmtQual', 'BsmtExposure',
    'FireplaceQu', 'GarageType', 'GarageFinish'
]

TEMPORAL_VARS = 'YearRemodAdd'

# this variable is to calculate the temporal variable,
# can be dropped afterwards
DROP_FEATURES = 'YrSold'

# variables to log transform
NUMERICALS_LOG_VARS = ['LotFrontage', '1stFlrSF', 'GrLivArea']

# numerical variables with NA in train set
NUMERICAL_VARS_WITH_NA = ['LotFrontage']

# categorical variables to encode
CATEGORICAL_VARS = ['MSZoning', 'Neighborhood', 'RoofStyle', 'MasVnrType',
                    'BsmtQual', 'BsmtExposure', 'HeatingQC', 'CentralAir',
                    'KitchenQual', 'FireplaceQu', 'GarageType', 'GarageFinish',
                    'PavedDrive']

DATA = pd.read_csv(r'C:\\Users\\DO01043\\Projects\\dami_deploy_ml_model\\datasets\\test.csv')

'''
We will create transfomer classes for preprocessing the data by inheriting from Scikit-learn BaseEstimator and TransformerMixin classes.
'''

class CategoricalImputer(BaseEstimator, TransformerMixin):
    """
    Categorical data missing value imputer
    """
    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'CategoricalImputer':
        """
        Fit statement to accomodate the sklearn pipeline, it returns the object [BaseEstimator] itself.
        """
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Apply the transforms to the dataframe
        """
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna('Missing')

        return X


if __name__ == "__main__":
    data = DATA
    transformed_data = CategoricalImputer(variables=CATEGORICAL_VARS_WITH_NA)
    print(transformed_data.fit(data))
    df = transformed_data.transform(data)
    export_csv = df.to_csv(r'C:\\Users\\DO01043\\Projects\\dami_deploy_ml_model\\datasets\\CategoricalImputer_data.csv')
    data = pd.read_csv(r'C:\\Users\\DO01043\\Projects\\dami_deploy_ml_model\\datasets\\CategoricalImputer_data.csv')
    for var in CATEGORICAL_VARS:
        print(var, np.round(data[var].isnull().mean(), 3) ,  ' % missing values')



class NumericalImputer(BaseEstimator, TransformerMixin):
    """
    Numerical mission value imputer
    """

    def __init__(self, variables=None):
        if  not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # Persist the mode in a dictionary
        self.imputer_dict_ = {}

        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].mode()[0]
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X


if __name__ == "__main__":
    data = DATA
    transformed_data = NumericalImputer(variables=NUMERICAL_VARS_WITH_NA)
    print(transformed_data.fit(data))
    df = transformed_data.transform(data)
    export_csv = df.to_csv(r'C:\\Users\\DO01043\\Projects\\dami_deploy_ml_model\\datasets\\NumericalImputer_data.csv')
    data = pd.read_csv(r'C:\\Users\\DO01043\\Projects\\dami_deploy_ml_model\\datasets\\NumericalImputer_data.csv')
    for var in NUMERICAL_VARS_WITH_NA:
        print(var, np.round(data[var].isnull().mean(), 3) ,  ' % missing values')



class TemporalVariableEstimator(BaseEstimator, TransformerMixin):
    """
    Temporal variable calculator (years)
    """
    def __init__(self, variables=None, reference_variable=None):
        if  not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        self.reference_variable = reference_variable

    def fit(self, X, y=None):
        # The step is needed to fit the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[self.reference_variable] - X[feature]
        return X

if __name__ == "__main__":
    transformed_data = TemporalVariableEstimator(variables=TEMPORAL_VARS, reference_variable=TEMPORAL_VARS)
    print(transformed_data.fit(data))
    df = transformed_data.transform(data)
    export_csv = df.to_csv(r'C:\\Users\\DO01043\\Projects\\dami_deploy_ml_model\\datasets\\TemporalVariableEstimator.csv')




class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    """Rare label categorical encoder"""

    def __init__(self, tol=0.05, variables=None):
        self.tol = tol
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # persist frequent labels in dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            # the encoder will learn the most frequent categories
            t = pd.Series(X[var].value_counts() / np.float(len(X)))
            # frequent labels:
            self.encoder_dict_[var] = list(t[t >= self.tol].index)

        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(X[feature].isin(
                self.encoder_dict_[feature]), X[feature], 'Rare')

        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """String to numbers categorical encoder."""

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y):
        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns) + ['target']

        # persist transforming dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            t = temp.groupby([var])['target'].mean().sort_values(
                ascending=True).index
            self.encoder_dict_[var] = {k: i for i, k in enumerate(t, 0)}

        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.encoder_dict_[feature])

        # check if transformer introduces NaN
        if X[self.variables].isnull().any().any():
            null_counts = X[self.variables].isnull().any()
            vars_ = {key: value for (key, value) in null_counts.items()
                     if value is True}
            raise ValueError(
                f'Categorical encoder has introduced NaN when '
                f'transforming categorical variables: {vars_.keys()}')

        return X


class LogTransformer(BaseEstimator, TransformerMixin):
    """Logarithm transformer."""

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # to accomodate the pipeline
        return self

    def transform(self, X):
        X = X.copy()

        # check that the values are non-negative for log transform
        if not (X[self.variables] > 0).all().all():
            vars_ = self.variables[(X[self.variables] <= 0).any()]
            raise ValueError(
                f"Variables contain zero or negative values, "
                f"can't apply log for vars: {vars_}")

        for feature in self.variables:
            X[feature] = np.log(X[feature])

        return X


class DropUnecessaryFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, variables_to_drop=None):
        self.variables = variables_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        X = X.drop(self.variables, axis=1)

        return X




















