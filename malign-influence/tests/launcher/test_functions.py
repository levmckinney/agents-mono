"""
Test functions for launch.py tests - defined at module level for proper pickling
"""

import time
from pathlib import Path

from pydantic import BaseModel


def dummy_entrypoint_function(args: BaseModel) -> None:
    """Module-level dummy function that can be properly pickled"""
    if hasattr(args, "name") and hasattr(args, "value"):
        print(f"Running job {getattr(args, 'name')} with value {getattr(args, 'value')}")
    else:
        print(f"Running job with args: {args}")


def integration_test_function(args: BaseModel) -> None:
    """Integration test function that sleeps and writes output"""
    # Type check to ensure we have the right args
    if not hasattr(args, "job_name") or not hasattr(args, "sleep_duration") or not hasattr(args, "output_file"):
        raise ValueError("Invalid arguments for test job")

    job_name = getattr(args, "job_name")
    sleep_duration = getattr(args, "sleep_duration")
    output_file = getattr(args, "output_file")

    print(f"Starting job: {job_name}")
    time.sleep(sleep_duration)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"Job {job_name} completed at {time.time()}")

    print(f"Completed job: {job_name}")


def long_running_test_function(args: BaseModel) -> None:
    """Test function that simulates a long-running job for signal testing"""
    name = getattr(args, "name")
    print(f"Starting long job: {name}")
    time.sleep(10)  # Long enough to be interrupted by signal
    print(f"Completed long job: {name}")


def slow_test_function(args: BaseModel) -> None:
    """Test function that sleeps for 5 seconds for signal testing"""
    name = getattr(args, "name")
    print(f"Starting slow job: {name}")
    time.sleep(5)  # Simulate work
    print(f"Completed slow job: {name}")


def failing_test_function(args: BaseModel) -> None:
    """Test function that raises an exception to test error handling"""
    name = getattr(args, "name")
    value = getattr(args, "value")
    print(f"Starting failing job: {name}")

    # Nested function call to create a deeper stack trace
    def nested_failure():
        def deeply_nested_failure():
            raise ValueError(f"Intentional test failure for job {name} with value {value}")

        deeply_nested_failure()

    nested_failure()


def torch_distributed_test_function(args: BaseModel) -> None:
    """Test function that verifies torch distributed environment is set up"""
    import os

    name = getattr(args, "name")
    output_file = getattr(args, "output_file")

    # Check for torch distributed environment variables
    rank = os.environ.get("RANK", "not_set")
    world_size = os.environ.get("WORLD_SIZE", "not_set")
    local_rank = os.environ.get("LOCAL_RANK", "not_set")
    master_addr = os.environ.get("MASTER_ADDR", "not_set")
    master_port = os.environ.get("MASTER_PORT", "not_set")

    output = f"Job: {name}\n"
    output += f"RANK: {rank}\n"
    output += f"WORLD_SIZE: {world_size}\n"
    output += f"LOCAL_RANK: {local_rank}\n"
    output += f"MASTER_ADDR: {master_addr}\n"
    output += f"MASTER_PORT: {master_port}\n"

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Append to file to capture all processes
    with open(output_path, "a") as f:
        f.write(output + "\n")
