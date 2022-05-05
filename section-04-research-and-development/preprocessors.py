from operator import index
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

# Watch out: we always inherit from BaseEstimator, TransformerMixin
class TemporalVariableTransformer(BaseEstimator, TransformerMixin):
    """Temporal elapsed time transformer."""

    def __init__(self, variables, reference_variable):
        
        # Check that variables is of type list!
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        
        # List of variables we want to computed the year difference for
        self.variables = variables
        # The reference year variable against which we compute the difference/elapsed time in years
        self.reference_variable = reference_variable

    def fit(self, X, y=None):
        # We don't learn anything, BUT
        # We need this step to fit the sklearn pipeline
        return self

    def transform(self, X):

        # Note that we copy X
        # So that we do not over-write the original dataframe
        X = X.copy()
        
        # This is our procedural code, packed in a class
        for feature in self.variables:
            X[feature] = X[self.reference_variable] - X[feature]

        return X

 
class Mapper(BaseEstimator, TransformerMixin):
    """Categorical missing value imputer."""

    # Variables that need the mapping, which mappings
    # variables: list
    # mappings: dictionary; however, that is not checked in this code
    def __init__(self, variables, mappings):

        # We check variables is a list
        # We could check that mappings is a dictionary, too!
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        self.variables = variables
        self.mappings = mappings

    def fit(self, X, y=None):
        # We don't learn anything from the data, BUT
        # We need the fit statement to accommodate the sklearn pipeline
        return self

    def transform(self, X):
        # This is the procedural piece of code, packed in a class!
        # Note that we copy X to avoid changing the original dataset
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.mappings)

        return X

class MeanImputer(BaseEstimator, TransformerMixin):
    """Numerical missing value imputer."""

    def __init__(self, variables):
        # Check that the variables are of type list
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        self.variables = variables

    def fit(self, X, y=None):
        # Learn and persist mean values in a dictionary
        self.imputer_dict_ = X[self.variables].mean().to_dict()
        return self

    def transform(self, X):
        # Note that we copy X to avoid changing the original dataset
        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X

class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    """Groups infrequent categories into a single string"""

    def __init__(self, variables, tol=0.05):
        # Check that the variables are of type list
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        
        # Rare if appears less than tol=5%
        self.tol = tol
        self.variables = variables
        self.replace_with = "Rare"

    def fit(self, X, y=None):
        # Learn persist frequent labels in dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            # Get the frequencies of each category/level
            t = pd.Series(X[var].value_counts(normalize=True))
            # Select frequent labels and save them - or infrequent?
            #freq_idx = t[t >= (self.tol)].index
            freq_idx = t[t < (self.tol)].index
            self.encoder_dict_[var] = list(freq_idx)
        
        return self

    def transform(self, X):
        # Note that we copy X to avoid changing the original dataset
        X = X.copy()
        for feature in self.variables:
            # Replace categories/levels that were detected as rare (fit) with label "Rare"
            X[feature] = np.where(X[feature].isin(self.encoder_dict_[feature]), X[feature], self.replace_with)

        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """String to numbers categorical encoder."""

    def __init__(self, variables):
        # Check that the variables are of type list
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        
        self.variables = variables

    def fit(self, X, y):
        # We want to order the labels according to their target values
        # Thus, we add target=y as column
        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns) + ["target"]

        # Learn and persist transforming dictionary
        self.encoder_dict_ = {}

        # Group-By using the variable (ans its categories)
        # and sort values according to their mean target
        for var in self.variables:
            t = temp.groupby([var])["target"].mean().sort_values(ascending=True).index
            # Save the label-index dictionary for each variable (key=label:value=index)
            self.encoder_dict_[var] = {k: i for i, k in enumerate(t, 0)}

        return self

    def transform(self, X):
        # Note that we copy X to avoid changing the original dataset
        X = X.copy()
        for feature in self.variables:
            # Encode labels: map label-index pairs (key=label:value=index)
            X[feature] = X[feature].map(self.encoder_dict_[feature])

        return X