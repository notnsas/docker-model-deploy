
# import requests
# import json

# # Ganti URL ini dengan endpoint Flask kamu
# url = "http://127.0.0.1:5000/api/prediction"  # contoh endpoint

# # Data contoh sesuai input API Fraud / transaksi
# payload = {
#     "location": "Jakarta",
#     "amount": 1500000,
#     # "jenis_transaksi": "transfer",
#     # "waktu": "23:45",
#     # "device_id": "abc123"
# }

# # Kirim request POST
# response = requests.post(url, json=payload)

# print("Status Code:", response.status_code)
# print("Response:")
# try:
#     print(response.json())
# except:
#     print(response.text)

# tests/test_api.py
import json
import pytest
from model_deployment import app

@pytest.fixture
def client():
    app.testing = True
    return app.test_client()

def test_api_predict_success(client):
    """Tes apakah endpoint /predict merespon dengan benar."""
    payload = {
        "location": "New York",
        "amount": 500000,
        # "jam_transaksi": 10,
        # "kategori_transaksi": 1
    }

    response = client.post(
        "/api/prediction",
        data=json.dumps(payload),
        content_type="application/json"
    )

    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data

def test_api_missing_fields(client):
    """Tes API harus memunculkan error jika input kurang."""
    payload = {
        "amount": 500000
        # field lain tidak diisi
    }

    response = client.post(
        "/api/prediction",
        data=json.dumps(payload),
        content_type="application/json"
    )

    assert response.status_code in (400, 422)