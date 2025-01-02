from wod_predictor.feature_engineering_parts.base import TransformerMixIn


class DropFeatures(TransformerMixIn):
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
        super().__init__()

    def fit(self, df):
        """
        identifies features with more than threshold percentage of missing values
        """
        self.features_to_drop = df.columns[df.isnull().mean() > self.threshold]
        super().fit()

    def transform(self, df):
        """
        Drops features previously identified as having more than threshold percentage of missing values
        """
        self.check_fit(df=df)
        safe_features_to_drop = set(self.features_to_drop).intersection(df.columns)
        return df.drop(columns=safe_features_to_drop)

    def fit_transform(self, df, threshold=0.9):
        self.fit(df, threshold)
        return self.transform(df)
