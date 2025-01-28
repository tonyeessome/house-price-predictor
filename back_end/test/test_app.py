import pytest
from fastapi.testclient import TestClient
from app import app, load_model, predict_price

# Create a test client for the FastAPI app
client = TestClient(app)

@pytest.fixture
def test_model():
    """Fixture to load the trained model for tests."""
    model = load_model("back_end/xgboost_retrained_model.json")
    assert model is not None
    return model

# Unit test 1: Test model loading
def test_load_model(test_model):
    """Test if the model loads correctly."""
    assert test_model.get_booster() is not None, "Model booster should not be None"

# Unit test 2: Test prediction function
def test_predict_price(test_model):
    """Test the prediction function with sample data."""
    sample_input = {
        "feature1": 3,
        "feature2": 2,
        "feature3": 1500,
        "feature4": 1
    }
    prediction = predict_price(test_model, sample_input)
    assert isinstance(prediction, float), "Prediction should return a float"
    assert prediction > 0, "Prediction should be greater than 0"

# Unit test 3: Test API endpoint for predictions
def test_api_predict_endpoint():
    """Test the /predict endpoint."""
    sample_input = {
        "feature1": 3,
        "feature2": 2,
        "feature3": 1500,
        "feature4": 1
    }
    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200, "API response should return 200"
    assert "prediction" in response.json(), "Response JSON should contain 'prediction'"
    assert response.json()["prediction"] > 0, "Prediction should be greater than 0"

if __name__ == "__main__":
    print("Running tests manually:")
    try:
        test_model_instance = load_model("back_end/xgboost_retrained_model.json")
        test_load_model(test_model_instance)
        print("Test 1 passed: Model loaded correctly.")
        test_predict_price(test_model_instance)
        print("Test 2 passed: Prediction function works correctly.")
        test_api_predict_endpoint()
        print("Test 3 passed: API endpoint works correctly.")
    except AssertionError as e:
        print(f"Test failed: {e}")