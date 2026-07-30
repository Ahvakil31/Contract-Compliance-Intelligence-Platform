import uuid
from locust import HttpUser, task, between

class LegalPlatformLoadTester(HttpUser):
    # Simulate realistic user delay: wait between 1 and 3 seconds between actions
    wait_time = between(1, 3)

    @task(1)
    def test_complete_ingestion_lifecycle(self):
        """Simulates an enterprise tenant uploading a contract and polling for completion."""
        correlation_id = f"LOAD-TEST-{uuid.uuid4().hex[:6].upper()}"
        headers = {"X-Correlation-ID": correlation_id}
        
        # Create a mock PDF memory stream payload
        payload_file = ("dummy_lease_2026.pdf", b"%PDF-1.4 mock content streams...", "application/pdf")
        
        # --- Step 1: Stress Post Upload Route ---
        with self.client.post(
            "/api/v1/contracts/upload", 
            files={"file": payload_file},
            headers=headers,
            catch_response=True
        ) as response:
            if response.status_code == 202:
                response_json = response.json()
                task_id = response_json.get("task_id")
                response.success()
            else:
                response.failure(f"Upload failed with HTTP status code: {response.status_code}")
                return

        # --- Step 2: Stress Status Polling Async Loop ---
        # Poll up to 5 times to simulate the frontend check interval
        for _ in range(5):
            with self.client.get(
                f"/api/v1/contracts/tasks/{task_id}",
                headers=headers,
                catch_response=True
            ) as poll_response:
                if poll_response.status_code == 200:
                    poll_json = poll_response.json()
                    status = poll_json.get("status")
                    
                    if status == "SUCCESS":
                        poll_response.success()
                        break
                    elif status == "FAILURE":
                        poll_response.failure("Background Celery pipeline failed processing.")
                        break
                    else:
                        # Task is still in PROGRESS or QUEUED state
                        poll_response.success()