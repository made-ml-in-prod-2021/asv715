REQUEST_URL = '/predict'


def test_can_call_endpoint(client):
    response = client.get(REQUEST_URL)

    assert response.status_code != 404, "Endpoint does not exist"


def test_can_return_validation_error_for_blank_data(client):
    response = client.get(REQUEST_URL)

    assert response.status_code == 400, "Endpoint does not return a validation error for blank data"


def test_can_return_validation_error_for_invalid_data(client, fake_invalid_data):
    request_data, request_features = fake_invalid_data

    response = client.get(
        REQUEST_URL,
        json={
            "data": request_data,
            "features": request_features
        },
    )

    assert response.status_code == 400, "Endpoint does not return a validation error for invalid data"


def test_can_get_correct_answer_with_valid_data(client, fake_valid_data):
    request_data, request_features = fake_valid_data

    response = client.get(
        REQUEST_URL,
        json={
            "data": request_data,
            "features": request_features
        },
    )

    assert response.status_code == 200, "Endpoint does not return a correct status for valid data"
    assert isinstance(response.json, list), "Response type is not list"
