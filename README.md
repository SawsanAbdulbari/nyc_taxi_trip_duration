# NYC Taxi Trip Duration Prediction

![Python](https://img.shields.io/badge/Python-3.9-blue.svg?style=flat&logo=python)
![Docker](https://img.shields.io/badge/Docker-19.03.12-blue.svg?style=flat&logo=docker)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0.2-orange.svg?style=flat&logo=scikit-learn)
![pandas](https://img.shields.io/badge/pandas-1.4.2-red.svg?style=flat&logo=pandas)
![numpy](https://img.shields.io/badge/numpy-1.22.3-lightblue.svg?style=flat&logo=numpy)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat&logo=open-source-initiative)
![GitHub repo size](https://img.shields.io/github/repo-size/yourusername/nyc-taxi-trip-duration?color=blue&logo=github)
![Type of Project](https://img.shields.io/badge/Type%20of%20Project-Machine%20Learning-orange?style=flat)
![Issues](https://img.shields.io/github/issues/yourusername/nyc-taxi-trip-duration)
![Forks](https://img.shields.io/github/forks/yourusername/nyc-taxi-trip-duration)
![Stars](https://img.shields.io/github/stars/yourusername/nyc-taxi-trip-duration)
![Views](https://views.whatilearened.today/views/github/yourusername/nyc-taxi-trip-duration.svg)
[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
  - [Using Docker](#using-docker)
  - [Local Setup](#local-setup)
- [Usage](#usage)
- [Data Description](#data-description)
- [Feature Engineering](#feature-engineering)
- [Model Details](#model-details)
- [Evaluation Metrics](#evaluation-metrics)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project aims to predict the trip duration of NYC taxi rides using machine learning techniques. The project involves data preprocessing, feature engineering, model building, and evaluation. The primary model used in this project is Ridge Regression.

![NYC Taxi](media/pixlr-image-generator-65d13dbf-1e0c-4096-958b-ee1aea9b399b.png)

## NYC Taxi App

### **[Click Here To Visit New York City Taxi Trip Duration Predictor App!](https://nyctaxi.streamlit.app/)**

This project contains code and resources for predicting model for NYC Taxi Trip Duration and a web-based dashboard built and hosted by Streamlit.

## Directory Structure

```
.
├── app
├── data
│   └── split
│       ├── train.csv
|       ├── test.csv.zip
│       └── val.csv
├── media
├── test
├── models
│   └── ridge_regression_model.pkl
├── notebooks
│   └── research.ipynb
├── scripts
│   ├── train.py
│   ├── utils.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── evaluation.py
│   └── load_model.py
├── report
│   └── Model Performance Report.pdf
├── Dockerfile
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md


```

## Installation

### Using Docker

1. **Build the Docker Image**

```sh
 docker build -t nyc-taxi-trip-duration .
```

2. **Run the Docker Container**

```sh
  docker run -it --rm nyc-taxi-trip-duration
```

### Local Setup

1. **Clone the Repository**

```sh
git clone https://github.com/SawsanAbdulbari/nyc_taxi_trip_duration.git
cd nyc-taxi-trip-duration
```

2. **Install Dependencies**

```sh
pip install -r requirements.txt
```

## Usage

### Run the Training Script

```sh
python scripts/train.py
```

### Load and Test the Model

```sh
python scripts/load_model.py
```

## Data Description

The dataset contains information about NYC taxi rides. Key columns include:

- `pickup_datetime`: The date and time when the trip started.
- `dropoff_datetime`: The date and time when the trip ended.
- `pickup_latitude`: The latitude of the pickup location.
- `pickup_longitude`: The longitude of the pickup location.
- `dropoff_latitude`: The latitude of the dropoff location.
- `dropoff_longitude`: The longitude of the dropoff location.
- `passenger_count`: The number of passengers in the vehicle.
- `trip_duration`: The duration of the trip in seconds.

## Feature Engineering

Various features are engineered to improve the model's performance:

- **Log Transformation**: Applied to trip duration to reduce skewness and stabilize variance, improving model robustness.
- **Time-Based Features**: Extracted from `pickup_datetime` (hour, day of the week, month, etc.).
- **Geographical Features**: Direction and Distances between pickup and dropoff locations (Haversine, Manhattan).
- **Trip Speed Feature**: Created from duration and distance to determine speed.
- **Airport Proximity Features**: Indicates if the pickup or dropoff locations are near major NYC airports.

## Model Details

The primary model used in this project is Ridge Regression. The model pipeline includes:

- **Column Transformer**: Applies `OneHotEncoder` to categorical features and `StandardScaler` to numeric features.
- **Ridge Regression**: A linear regression model with L2 regularization to prevent overfitting.

## Evaluation Metrics

The model is evaluated using the following metrics:

- **Root Mean Squared Error (RMSE)**: Measures the average magnitude of errors.
- **R² Score**: Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.

## Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
