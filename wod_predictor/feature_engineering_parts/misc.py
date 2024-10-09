class DropFeatures:
    """
    Identifies and drops features with more than threshold percentage of missing values

    Parameters:
    ----------
    threshold: float
        threshold percentage of missing values to make feature "dropable"
    """

    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.features_to_drop = []

    def fit(self, df):
        """
        identifies features with more than threshold percentage of missing values
        """
        self.features_to_drop = df.columns[df.isnull().mean() > self.threshold]

    def transform(self, df):
        """
        Drops features previously identified as having more than threshold percentage of missing values
        """
        safe_features_to_drop = set(self.features_to_drop).intersection(df.columns)
        return df.drop(columns=safe_features_to_drop)

    def fit_transform(self, df, threshold=0.9):
        self.fit(df, threshold)
        return self.transform(df)
