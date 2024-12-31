# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- **Added:** Description of new features.
- **Changed:** Description of changes in existing functionality.
- **Deprecated:** Description of soon-to-be-removed features.
- **Removed:** Description of now removed features.
- **Fixed:** Description of bug fixes.
- **Security:** Description of security improvements.

## [1.0.0] - 2024-12-31
### Added
- Workout description encoding as vector embeddings
- Scaling data using minmax, normalization or percentiles
- Hyper parameter tuning infrastructure using Optuna
- Neural Network Implementation
- Athlete Info Feature Engineering: new columns for age, height, weight and gender
- Option to remove scaled workouts
- Option to drop columns with number of missing values higher than a specific threshold
- Model comparison with a "Benchmark" to see performance lift in terms of MAE
- New preprocessed methods to convert mixed data types into Reps per minute or simply to expected time.

## [0.2.0] - 2024-07-05
### Added
- Seperate `DataSplitter` class to handle data splitting early on

### Changed
- Big refactor of codebase to prevent data leakage. Preprocessing and FE pipelines now work with `fit` and `transform` steps

### Removed
- Removed all splitting functionality from the models, now it lies in the `DataSplitter` class.
## [0.1.0] - 2024-06-15

### Added
- Initial release.
