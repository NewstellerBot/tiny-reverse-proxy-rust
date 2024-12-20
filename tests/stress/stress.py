import requests
import time
from concurrent.futures import ThreadPoolExecutor


def measure_request_time(url):
    try:
        start_time = time.perf_counter()
        response = requests.get(url)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        return elapsed_time, response.status_code
    except requests.exceptions.RequestException as e:
        return None, str(e)


def test_concurrent_requests(url, num_requests, max_workers=10):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(measure_request_time, url) for _ in range(num_requests)
        ]
        results = [future.result() for future in futures]
    return results


# URL to test
url_to_test = "http://localhost:8000"

# Number of requests and concurrency level
num_requests = 7000
max_workers = 32

# Run the test
print(f"Testing {num_requests} requests to {url_to_test} with {max_workers} workers...")
results = test_concurrent_requests(url_to_test, num_requests, max_workers)

# Analyze results
successful_requests = [res for res in results if res[0] is not None]
failed_requests = [res for res in results if res[0] is None]

# Report results
print("\nResults:")
print(f"Total Requests: {num_requests}")
print(f"Successful Requests: {len(successful_requests)}")
print(f"Failed Requests: {len(failed_requests)}")

if successful_requests:
    times = [res[0] for res in successful_requests]
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    print(f"Average Time: {avg_time:.4f} seconds")
    print(f"Min Time: {min_time:.4f} seconds")
    print(f"Max Time: {max_time:.4f} seconds")
    print(
        ";".join(
            list(
                map(
                    str,
                    [
                        num_requests,
                        len(successful_requests),
                        avg_time,
                        min_time,
                        max_time,
                    ],
                )
            )
        )
    )
else:
    print("All requests failed.")

if failed_requests:
    print("\nFailure Details:")
    error_counts = {}
    for _, error in failed_requests:
        error_counts[error] = error_counts.get(error, 0) + 1
    for error, count in error_counts.items():
        print(f"{count} requests failed with error: {error}")
