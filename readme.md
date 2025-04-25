# California Housing Prediction API

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue) ![License](https://img.shields.io/github/license/jaymwangi/california-housing-api) [![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

A FastAPI-based service that predicts California housing prices using a trained regression model. Built on the `sklearn.datasets.fetch_california_housing` dataset.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation & Setup](#installation-setup)
- [API Endpoints](#api-endpoints)
- [Deployment](#deployment)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Overview
This API serves predictions for median California house prices using key features like income, room count, and location.

## Features
- **Predict housing prices** with 5 input features.
- **RESTful API** with FastAPI (Swagger UI included at `/docs`).
- **Lightweight & optimized** for free-tier hosting (works locally, but requires a paid plan for production on Render due to memory constraints).
- **Validation** for input data types and ranges.

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip

### Steps
1. Clone the repo:
   ```bash
   git clone https://github.com/jaymwangi/california-housing-api.git
   cd california-housing-api
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the `model/california_housing_model.pkl` is already saved and placed in the `model/` directory. (The model has been pre-trained.)

4. Start the server:
   ```bash
   uvicorn app:app --reload
   ```

Access the API at `http://localhost:8000` or explore endpoints at `http://localhost:8000/docs`.

## API Endpoints

### `POST /predict`
Predicts the median house price.

**Request**:
```json
{
  "MedInc": 3.1,
  "AveRooms": 4.2,
  "AveOccup": 3.4,
  "Latitude": 37.77,
  "Longitude": -122.42
}
```

**Response**:
```json
{
  "predicted_price": 3.45
}
```

**Errors**:
- `422 Unprocessable Entity` for invalid input (e.g., non-numeric values).

## Deployment

### Deploy to Render
1. Click the **Deploy to Render** button above.
2. Set environment variables (if needed).
3. Ensure the `model/california_housing_model.pkl` file is included in your deployment.

**Memory Considerations**:  
The model works perfectly on local machines, but the free-tier Render plan may run out of memory during production. A paid plan is recommended for production deployment.

## Testing
Run unit and integration tests:
```bash
pytest tests/ -v
```

Tests cover:
- Model prediction consistency
- API endpoint validation
- Error handling

## Contributing
1. Open an issue to discuss your proposed change.
2. Fork the repo and create a feature branch.
3. Add tests for new functionality.
4. Submit a pull request with a detailed description.

## License
MIT License. See [LICENSE](LICENSE).
