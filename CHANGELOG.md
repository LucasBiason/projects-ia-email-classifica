# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2025-01-27

### Changed
- Migrated from requirements.txt to Poetry for dependency management

## [1.0.0] - 2024-12-19

### Added
- **Email Classification API** with FastAPI
- **Machine Learning Model** using Multinomial Naive Bayes
- **Text Processing** with CountVectorizer
- **RESTful Endpoints**:
  - `GET /` - Service status
  - `GET /health` - Health check
  - `POST /predict` - Email classification
- **Data Validation** with Pydantic schemas
- **Comprehensive Testing** with 100% code coverage
- **Docker Support** for easy deployment
- **Code Quality** with type hints and docstrings

### Changed
- Updated dependencies to latest compatible versions
- Improved Docker configuration for Python 3.11
- Enhanced test structure and coverage

### Fixed
- Resolved dependency conflicts and version issues
- Fixed Docker build and permission problems
- Corrected test failures and API response format
