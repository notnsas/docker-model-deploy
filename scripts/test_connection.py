
import requests
import json

class MobileConnectionTester:
    def __init__(self, base_url):
        """
        base_url = URL API Anda, contoh:
        http://192.168.1.10:5000
        http://10.0.2.2:5000  <-- Android emulator
        """
        self.base_url = base_url.rstrip("/")

    def test_connection(self):
        """Cek apakah server API hidup."""
        url = f"{self.base_url}/ping"
        try:
            response = requests.get(url, timeout=5)
            return {
                "status": "success",
                "server_response": response.json()
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def test_fraud_prediction(self, location, amount):
        """Kirim transaksi ke server untuk dianalisis."""
        url = f"{self.base_url}/api/prediction"
        payload = {
            "location": location,
            "amount": amount,
            # "transaction_type": ttype,
            # "time": time
        }

        try:
            response = requests.post(url, json=payload, timeout=7)
            return response.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}

# Example penggunaan
if __name__ == "__main__":
    tester = MobileConnectionTester("http://127.0.0.1:5000")

    print("\n=== TEST SERVER CONNECTION ===")
    print(tester.test_connection())

    print("\n=== TEST FRAUD PREDICTION ===")
    print(
        tester.test_fraud_prediction(
            location="jakarta",
            amount=5000000,
            # ttype="payment",
            # time="normal"
        )
    )