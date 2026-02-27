import time

from locust import HttpUser, between, task


class QuantumFoldUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict_structure(self):
        sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"
        response = self.client.post("/predict", json={"sequence": sequence, "model_type": "quantum"})
        job_id = response.json()["job_id"]
        while True:
            status = self.client.get(f"/status/{job_id}")
            if status.json()["status"] == "completed":
                self.client.get(f"/result/{job_id}")
                break
            time.sleep(2)
