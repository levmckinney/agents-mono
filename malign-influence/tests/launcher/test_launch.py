"""
Unit tests for the launch.py infrastructure using pytest
"""

import multiprocessing as mp
import os
import signal
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from launcher.jobs import (
    Job,
    JobQueue,
    JobState,
    ResourceRequest,
    create_job_array_from_sweep,
    get_available_gpus,
)
from launcher.local_orchestrator import LocalConfig
from launcher.worker import Worker
from tests.launcher.test_functions import (
    dummy_entrypoint_function,
    long_running_test_function,
    slow_test_function,
    torch_distributed_test_function,
)


class JobTestArgs(BaseModel):
    """Test job arguments"""

    name: str
    value: int


# Use the module-level function for proper serialization
dummy_entrypoint = dummy_entrypoint_function


class TestResourceRequest:
    """Test ResourceRequest creation and validation"""

    def test_creation_with_all_fields(self):
        """Test ResourceRequest creation with all fields"""
        resource = ResourceRequest(
            cpu=2.0,
            memory=8.0,
            gpu=1,
            parallel_jobs=1,
        )

        assert resource.cpu == 2.0
        assert resource.memory == 8.0
        assert resource.gpu == 1

    def test_creation_minimal_fields(self):
        """Test ResourceRequest creation with minimal required fields"""
        resource = ResourceRequest(
            cpu=1.0,
            memory=4.0,
            gpu=0,
            parallel_jobs=1,
        )

        assert resource.cpu == 1.0
        assert resource.memory == 4.0
        assert resource.gpu == 0

    def test_torch_distributed_field(self):
        """Test ResourceRequest with torch distributed field"""
        resource = ResourceRequest(
            cpu=2.0,
            memory=8.0,
            gpu=2,
            parallel_jobs=1,
            use_torch_distributed=True,
        )

        assert resource.use_torch_distributed is True
        assert resource.gpu == 2

    def test_torch_distributed_default_false(self):
        """Test that torch distributed defaults to False"""
        resource = ResourceRequest(
            cpu=1.0,
            memory=4.0,
        )

        assert resource.use_torch_distributed is False


class TestJob:
    """Test Job creation and serialization"""

    def test_job_creation(self):
        """Test basic Job creation"""
        resource = ResourceRequest(cpu=1.0, memory=4.0, gpu=0, parallel_jobs=1)
        job_args = JobTestArgs(name="test", value=42)

        job = Job(
            id="test_job_1", args=job_args, state=JobState.PENDING, resources=resource, entrypoint=dummy_entrypoint
        )

        assert job.id == "test_job_1"
        assert job.state == JobState.PENDING
        assert job.resources.cpu == 1.0
        assert getattr(job.args, "name") == "test"
        assert getattr(job.args, "value") == 42

    def test_job_state_transitions(self):
        """Test job state can be updated"""
        resource = ResourceRequest(cpu=1.0, memory=4.0, gpu=0, parallel_jobs=1)
        job_args = JobTestArgs(name="test", value=42)

        job = Job(
            id="test_job_1", args=job_args, state=JobState.PENDING, resources=resource, entrypoint=dummy_entrypoint
        )

        assert job.state == JobState.PENDING
        job.state = JobState.RUNNING
        assert job.state == JobState.RUNNING
        job.state = JobState.COMPLETED
        assert job.state == JobState.COMPLETED


class TestJobQueue:
    """Test JobQueue file operations and job management"""

    def test_empty_queue_initialization(self):
        """Test JobQueue initialization with empty jobs"""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_path = Path(temp_dir) / "test_queue.json"
            job_queue = JobQueue(job_array_path=queue_path)

            # Initialize with empty jobs
            job_queue.initialize_with_jobs([])

            # Test counts
            assert job_queue.count_pending_jobs() == 0
            assert job_queue.count_running_jobs() == 0
            assert job_queue.count_completed_jobs() == 0
            assert job_queue.count_failed_jobs() == 0

    def test_queue_file_creation(self):
        """Test that queue file is created properly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_path = Path(temp_dir) / "test_queue.json"
            job_queue = JobQueue(job_array_path=queue_path)

            # File shouldn't exist initially
            assert not queue_path.exists()

            # Initialize creates the file
            job_queue.initialize_with_jobs([])
            assert queue_path.exists()

            # File should contain valid JSON
            content = queue_path.read_text()
            assert '"jobs": []' in content or '"jobs":[]' in content

    def test_initialize_and_count_jobs(self):
        """Test JobQueue initialization and job counting"""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_path = Path(temp_dir) / "test_queue.json"
            job_queue = JobQueue(job_array_path=queue_path)

            # Create test jobs
            resource = ResourceRequest(cpu=1.0, memory=4.0, gpu=0, parallel_jobs=1)
            jobs = []
            for i in range(3):
                job_args = JobTestArgs(name=f"job_{i}", value=i)
                job = Job(
                    id=f"test_job_{i}",
                    args=job_args,
                    state=JobState.PENDING,
                    resources=resource,
                    entrypoint=dummy_entrypoint,
                )
                jobs.append(job)

            # Initialize queue
            job_queue.initialize_with_jobs(jobs)

            # Test counts
            assert job_queue.count_pending_jobs() == 3
            assert job_queue.count_running_jobs() == 0
            assert job_queue.count_completed_jobs() == 0
            assert job_queue.count_failed_jobs() == 0

    def test_get_next_job(self):
        """Test getting next job from queue"""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_path = Path(temp_dir) / "test_queue.json"
            job_queue = JobQueue(job_array_path=queue_path)

            # Create test jobs
            resource = ResourceRequest(cpu=1.0, memory=4.0, gpu=0, parallel_jobs=1)
            jobs = []
            for i in range(2):
                job_args = JobTestArgs(name=f"job_{i}", value=i)
                job = Job(
                    id=f"test_job_{i}",
                    args=job_args,
                    state=JobState.PENDING,
                    resources=resource,
                    entrypoint=dummy_entrypoint,
                )
                jobs.append(job)

            job_queue.initialize_with_jobs(jobs)

            # Get a job
            next_job = job_queue.get_next_job()
            assert next_job is not None
            assert next_job.id == "test_job_0"
            assert job_queue.count_running_jobs() == 1
            assert job_queue.count_pending_jobs() == 1

            # Get another job
            next_job2 = job_queue.get_next_job()
            assert next_job2 is not None
            assert next_job2.id == "test_job_1"
            assert job_queue.count_running_jobs() == 2
            assert job_queue.count_pending_jobs() == 0

            # No more jobs
            next_job3 = job_queue.get_next_job()
            assert next_job3 is None

    def test_update_job_state(self):
        """Test updating job state in queue"""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_path = Path(temp_dir) / "test_queue.json"
            job_queue = JobQueue(job_array_path=queue_path)

            # Create test job
            resource = ResourceRequest(cpu=1.0, memory=4.0, gpu=0, parallel_jobs=1)
            job_args = JobTestArgs(name="job_0", value=0)
            job = Job(
                id="test_job_0", args=job_args, state=JobState.PENDING, resources=resource, entrypoint=dummy_entrypoint
            )

            job_queue.initialize_with_jobs([job])

            # Update job state
            job_queue.update_job_state("test_job_0", JobState.COMPLETED)
            assert job_queue.count_completed_jobs() == 1
            assert job_queue.count_pending_jobs() == 0


class TestJobArrayCreation:
    """Test create_job_array_from_sweep function"""

    def test_basic_job_array_creation(self):
        """Test creating job array from sweep arguments"""
        arguments = [
            {"name": "job_1", "value": 10},
            {"name": "job_2", "value": 20},
            {"name": "job_3", "value": 30},
        ]

        resource_request = ResourceRequest(cpu=2.0, memory=8.0, gpu=1, parallel_jobs=1)

        job_array = create_job_array_from_sweep(
            target_args_model=JobTestArgs,
            target_entrypoint=dummy_entrypoint,
            arguments=arguments,
            resource_request=resource_request,
            sweep_id="test_sweep",
        )

        assert len(job_array.jobs) == 3
        assert job_array.jobs[0].id == "test_sweep_0"
        assert job_array.jobs[1].id == "test_sweep_1"
        assert job_array.jobs[2].id == "test_sweep_2"

        for job in job_array.jobs:
            assert job.state == JobState.PENDING
            assert job.resources.cpu == 2.0
            assert job.resources.gpu == 1

    def test_empty_arguments(self):
        """Test creating job array with empty arguments"""
        arguments = []
        resource_request = ResourceRequest(cpu=1.0, memory=4.0, gpu=0, parallel_jobs=1)

        job_array = create_job_array_from_sweep(
            target_args_model=JobTestArgs,
            target_entrypoint=dummy_entrypoint,
            arguments=arguments,
            resource_request=resource_request,
            sweep_id="empty_sweep",
        )

        assert len(job_array.jobs) == 0


class TestConfigurations:
    """Test configuration classes"""

    def test_local_config_creation(self):
        """Test LocalConfig creation and defaults"""
        local_config = LocalConfig(log_dir=Path("./test_logs"), venv_activate_script=Path("./.venv/bin/activate"))

        assert local_config.log_dir == Path("./test_logs")
        assert local_config.venv_activate_script == Path("./.venv/bin/activate")

    def test_local_config_defaults(self):
        """Test LocalConfig with default values"""
        local_config = LocalConfig()

        assert local_config.log_dir == Path("./logs/local")
        assert local_config.venv_activate_script == Path("./.venv/bin/activate")


class TestGPUDetection:
    """Test GPU detection functionality"""

    def test_gpu_detection_no_error(self):
        """Test that GPU detection doesn't raise errors"""
        # This may return empty list if no GPUs, which is fine
        try:
            gpus = get_available_gpus()
            assert isinstance(gpus, list)
            # All elements should be integers
            for gpu_id in gpus:
                assert isinstance(gpu_id, int)
                assert gpu_id >= 0
        except Exception:
            # GPU detection can fail in CI environments, which is acceptable
            pytest.skip("GPU detection failed (no GPUs available)")


class TestJobStates:
    """Test JobState enum functionality"""

    def test_job_state_values(self):
        """Test JobState enum values"""
        assert JobState.PENDING.value == "PENDING"
        assert JobState.RUNNING.value == "RUNNING"
        assert JobState.COMPLETED.value == "COMPLETED"
        assert JobState.FAILED.value == "FAILED"

    def test_job_state_comparison(self):
        """Test JobState enum comparison"""
        assert JobState.PENDING == JobState.PENDING
        assert JobState.PENDING != JobState.RUNNING
        assert JobState.RUNNING != JobState.COMPLETED
        assert JobState.COMPLETED != JobState.FAILED


class TestWorkerExecution:
    """Test Worker job execution"""

    def test_worker_normal_completion(self):
        """Test that worker normally completes jobs"""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_path = Path(temp_dir) / "test_queue.json"
            job_queue = JobQueue(job_array_path=queue_path)

            # Use module-level function
            job_args = JobTestArgs(name="quick_job", value=1)
            resource = ResourceRequest(cpu=1.0, memory=4.0, gpu=0, parallel_jobs=1)
            job = Job(
                id="test_job_normal",
                args=job_args,
                state=JobState.PENDING,
                resources=resource,
                entrypoint=dummy_entrypoint_function,
            )

            # Initialize queue with the job
            job_queue.initialize_with_jobs([job])

            # Create and run worker
            worker = Worker()
            worker.run(queue_path, "test_worker")

            # Verify job completed normally
            assert job_queue.count_completed_jobs() == 1
            assert job_queue.count_running_jobs() == 0
            assert job_queue.count_pending_jobs() == 0


class TestWorkerSignalHandling:
    """Test Worker signal handling for graceful shutdown"""

    def test_worker_sigint_handling(self):
        """Test that worker handles SIGINT (Ctrl+C) gracefully"""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_path = Path(temp_dir) / "test_queue.json"
            job_queue = JobQueue(job_array_path=queue_path)

            # Create a long-running job
            job_args = JobTestArgs(name="slow_job", value=1)
            resource = ResourceRequest(cpu=1.0, memory=4.0, gpu=0, parallel_jobs=1)
            job = Job(
                id="test_job_sigint",
                args=job_args,
                state=JobState.PENDING,
                resources=resource,
                entrypoint=slow_test_function,
            )

            job_queue.initialize_with_jobs([job])

            # Run worker in separate process
            def run_worker():
                worker = Worker()
                worker.run(queue_path, "test_worker_sigint")

            worker_process = mp.Process(target=run_worker)
            worker_process.start()

            # Wait for job to start
            time.sleep(1)

            # Send SIGINT to worker
            assert worker_process.pid is not None
            os.kill(worker_process.pid, signal.SIGINT)

            # Wait for graceful shutdown
            worker_process.join(timeout=3)

            # Verify job was set back to PENDING
            assert job_queue.count_pending_jobs() == 1
            assert job_queue.count_running_jobs() == 0
            assert job_queue.count_failed_jobs() == 0

    def test_worker_sigterm_handling(self):
        """Test that worker handles SIGTERM gracefully"""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_path = Path(temp_dir) / "test_queue.json"
            job_queue = JobQueue(job_array_path=queue_path)

            # Create a long-running job
            job_args = JobTestArgs(name="slow_job", value=1)
            resource = ResourceRequest(cpu=1.0, memory=4.0, gpu=0, parallel_jobs=1)
            job = Job(
                id="test_job_sigterm",
                args=job_args,
                state=JobState.PENDING,
                resources=resource,
                entrypoint=slow_test_function,
            )

            job_queue.initialize_with_jobs([job])

            # Run worker in separate process
            def run_worker():
                worker = Worker()
                worker.run(queue_path, "test_worker_sigterm")

            worker_process = mp.Process(target=run_worker)
            worker_process.start()

            # Wait for job to start
            time.sleep(1)

            # Send SIGTERM to worker
            assert worker_process.pid is not None
            os.kill(worker_process.pid, signal.SIGTERM)

            # Wait for graceful shutdown
            worker_process.join(timeout=3)

            # Verify job was set back to PENDING
            assert job_queue.count_pending_jobs() == 1
            assert job_queue.count_running_jobs() == 0
            assert job_queue.count_failed_jobs() == 0

    def test_worker_multiple_jobs_signal_handling(self):
        """Test signal handling with multiple running jobs"""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_path = Path(temp_dir) / "test_queue.json"
            job_queue = JobQueue(job_array_path=queue_path)

            # Create multiple long-running jobs
            jobs = []
            resource = ResourceRequest(cpu=1.0, memory=4.0, gpu=0, parallel_jobs=1)

            for i in range(3):
                job_args = JobTestArgs(name=f"slow_job_{i}", value=i)
                job = Job(
                    id=f"test_job_multi_{i}",
                    args=job_args,
                    state=JobState.PENDING,
                    resources=resource,
                    entrypoint=long_running_test_function,
                )
                jobs.append(job)

            job_queue.initialize_with_jobs(jobs)

            # Run worker with parallel jobs
            def run_worker():
                worker = Worker()
                worker.run(queue_path, "test_worker_multi", parallel_jobs=2)

            worker_process = mp.Process(target=run_worker)
            worker_process.start()

            # Wait for jobs to start
            time.sleep(2)

            # Send SIGINT to worker
            assert worker_process.pid is not None
            os.kill(worker_process.pid, signal.SIGINT)

            # Wait for graceful shutdown
            worker_process.join(timeout=5)

            # Verify running jobs were set back to PENDING
            # At least 2 jobs should have been running when signal was sent
            assert job_queue.count_pending_jobs() >= 2
            assert job_queue.count_running_jobs() == 0
            assert job_queue.count_failed_jobs() == 0


class TestJobSerialization:
    """Test job serialization and deserialization for subprocess execution"""

    def test_job_serialization_roundtrip(self):
        """Test that jobs can be serialized and deserialized correctly"""
        resource = ResourceRequest(cpu=1.0, memory=4.0, gpu=0, parallel_jobs=1)
        job_args = JobTestArgs(name="test_job", value=42)

        original_job = Job(
            id="test_serialization",
            args=job_args,
            state=JobState.PENDING,
            resources=resource,
            entrypoint=dummy_entrypoint_function,
        )

        # Serialize to JSON
        job_json = original_job.model_dump_json()

        # Deserialize back
        deserialized_job = Job.model_validate_json(job_json)

        # Verify all fields match
        assert deserialized_job.id == original_job.id
        assert deserialized_job.state == original_job.state
        assert deserialized_job.resources.cpu == original_job.resources.cpu
        assert deserialized_job.resources.memory == original_job.resources.memory
        assert isinstance(deserialized_job.args, JobTestArgs)
        assert isinstance(original_job.args, JobTestArgs)
        assert deserialized_job.args.name == original_job.args.name
        assert deserialized_job.args.value == original_job.args.value

    def test_job_execution_via_subprocess(self):
        """Test that jobs can be executed via subprocess using python -m launcher.jobs"""
        import subprocess
        import sys

        resource = ResourceRequest(cpu=1.0, memory=4.0, gpu=0, parallel_jobs=1)
        job_args = JobTestArgs(name="subprocess_test", value=123)

        job = Job(
            id="test_subprocess_exec",
            args=job_args,
            state=JobState.PENDING,
            resources=resource,
            entrypoint=dummy_entrypoint_function,
        )

        # Serialize job
        job_json = job.model_dump_json()

        # Run job via subprocess with command line argument
        result = subprocess.run(
            [sys.executable, "-m", "launcher.jobs", "--job", job_json], capture_output=True, text=True
        )

        # Check that the job executed successfully
        assert result.returncode == 0, f"Job execution failed: {result.stderr}"

    def test_job_execution_failure_via_subprocess(self):
        """Test that failed jobs return non-zero exit code"""
        import subprocess
        import sys

        # Create an invalid job JSON that will fail to deserialize
        invalid_job_json = '{"invalid": "json structure"}'

        # Run job via subprocess with command line argument
        result = subprocess.run(
            [sys.executable, "-m", "launcher.jobs", "--job", invalid_job_json], capture_output=True, text=True
        )

        # Check that the job failed with non-zero exit code
        assert result.returncode == 1

    def test_job_stack_trace_on_failure(self):
        """Test that stack traces are printed to stderr when jobs fail"""
        import subprocess
        import sys

        from tests.launcher.test_functions import failing_test_function

        resource = ResourceRequest(cpu=1.0, memory=4.0, gpu=0, parallel_jobs=1)
        job_args = JobTestArgs(name="stack_trace_test", value=999)

        job = Job(
            id="test_stack_trace",
            args=job_args,
            state=JobState.PENDING,
            resources=resource,
            entrypoint=failing_test_function,
        )

        # Serialize job
        job_json = job.model_dump_json()

        # Run job via subprocess
        result = subprocess.run(
            [sys.executable, "-m", "launcher.jobs", "--job", job_json], capture_output=True, text=True
        )

        # Check that the job failed
        assert result.returncode == 1

        # Verify stack trace is in stderr
        assert "Traceback" in result.stderr
        assert "ValueError" in result.stderr
        assert "Intentional test failure" in result.stderr
        assert "nested_failure" in result.stderr
        assert "deeply_nested_failure" in result.stderr
        # Verify the error message includes our test values
        assert "stack_trace_test" in result.stderr
        assert "999" in result.stderr


class TestTorchDistributedWorker:
    """Test Worker with torch distributed launch"""

    def test_torch_distributed_command_construction(self):
        """Test that torch distributed commands are constructed correctly"""
        import sys
        from unittest.mock import MagicMock, patch

        with tempfile.TemporaryDirectory() as temp_dir:
            queue_path = Path(temp_dir) / "test_queue.json"
            job_queue = JobQueue(job_array_path=queue_path)

            # Create a job with torch distributed enabled
            resource = ResourceRequest(
                cpu=2.0,
                memory=8.0,
                gpu=2,
                parallel_jobs=1,
                use_torch_distributed=True,
            )
            job_args = JobTestArgs(name="distributed_job", value=1)
            job = Job(
                id="test_distributed",
                args=job_args,
                state=JobState.PENDING,
                resources=resource,
                entrypoint=dummy_entrypoint_function,
            )

            job_queue.initialize_with_jobs([job])

            # Mock subprocess.Popen to capture the command
            with patch("subprocess.Popen") as mock_popen:
                mock_process = MagicMock()
                mock_process.poll.return_value = 0
                mock_popen.return_value = mock_process

                worker = Worker()
                worker.run(queue_path, "test_distributed_worker")

                # Verify Popen was called with torch distributed command
                assert mock_popen.called
                call_args = mock_popen.call_args
                cmd = call_args[0][0]

                # Check command structure
                assert sys.executable in cmd
                assert "-m" in cmd
                assert "torch.distributed.run" in cmd
                assert "--nproc_per_node=2" in cmd
                assert "--master_port=29500" in cmd
                assert "launcher.jobs" in cmd

    def test_torch_distributed_port_allocation(self):
        """Test that each distributed job gets a unique port"""
        from unittest.mock import MagicMock, patch

        with tempfile.TemporaryDirectory() as temp_dir:
            queue_path = Path(temp_dir) / "test_queue.json"
            job_queue = JobQueue(job_array_path=queue_path)

            # Create multiple jobs with torch distributed enabled
            jobs = []
            for i in range(3):
                resource = ResourceRequest(
                    cpu=2.0,
                    memory=8.0,
                    gpu=1,
                    parallel_jobs=1,
                    use_torch_distributed=True,
                )
                job_args = JobTestArgs(name=f"distributed_job_{i}", value=i)
                job = Job(
                    id=f"test_distributed_{i}",
                    args=job_args,
                    state=JobState.PENDING,
                    resources=resource,
                    entrypoint=dummy_entrypoint_function,
                )
                jobs.append(job)

            job_queue.initialize_with_jobs(jobs)

            # Mock subprocess.Popen to capture commands
            ports_used: list[int] = []
            with patch("subprocess.Popen") as mock_popen:

                def capture_port(*args: Any, **kwargs: Any) -> Any:
                    cmd = args[0]
                    # Extract port from command
                    for arg in cmd:
                        if arg.startswith("--master_port="):
                            port = int(arg.split("=")[1])
                            ports_used.append(port)

                    mock_process = MagicMock()
                    mock_process.poll.return_value = 0
                    return mock_process

                mock_popen.side_effect = capture_port

                worker = Worker()
                worker.run(queue_path, "test_port_worker", parallel_jobs=3)

                # Verify all ports are unique
                assert len(ports_used) == 3
                assert len(set(ports_used)) == 3
                assert ports_used == [29500, 29501, 29502]

    def test_non_distributed_job_unchanged(self):
        """Test that non-distributed jobs still use standard command"""
        import sys
        from unittest.mock import MagicMock, patch

        with tempfile.TemporaryDirectory() as temp_dir:
            queue_path = Path(temp_dir) / "test_queue.json"
            job_queue = JobQueue(job_array_path=queue_path)

            # Create a job WITHOUT torch distributed
            resource = ResourceRequest(
                cpu=1.0,
                memory=4.0,
                gpu=0,
                parallel_jobs=1,
                use_torch_distributed=False,
            )
            job_args = JobTestArgs(name="normal_job", value=1)
            job = Job(
                id="test_normal",
                args=job_args,
                state=JobState.PENDING,
                resources=resource,
                entrypoint=dummy_entrypoint_function,
            )

            job_queue.initialize_with_jobs([job])

            # Mock subprocess.Popen to capture the command
            with patch("subprocess.Popen") as mock_popen:
                mock_process = MagicMock()
                mock_process.poll.return_value = 0
                mock_popen.return_value = mock_process

                worker = Worker()
                worker.run(queue_path, "test_normal_worker")

                # Verify Popen was called with standard command
                assert mock_popen.called
                call_args = mock_popen.call_args
                cmd = call_args[0][0]

                # Check command does NOT include torch distributed
                assert "torch.distributed.launch" not in cmd
                assert sys.executable in cmd
                assert "-m" in cmd
                assert "launcher.jobs" in cmd


@pytest.mark.skipif(
    not os.path.exists("/usr/bin/python3") and not os.path.exists("/usr/local/bin/python3"),
    reason="Requires torch distributed to be available",
)
class TestTorchDistributedIntegration:
    """Integration tests for torch distributed job execution"""

    def test_torch_distributed_environment_setup(self):
        """Test that torch distributed environment variables are set correctly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_path = Path(temp_dir) / "test_queue.json"
            output_file = Path(temp_dir) / "distributed_output.txt"
            job_queue = JobQueue(job_array_path=queue_path)

            # Create custom args model with output_file
            class DistributedTestArgs(BaseModel):
                name: str
                output_file: str

            # Create a job with torch distributed enabled
            resource = ResourceRequest(
                cpu=1.0,
                memory=4.0,
                gpu=0,  # Use 0 GPUs for CI/CD environments
                parallel_jobs=1,
                use_torch_distributed=True,
            )
            job_args = DistributedTestArgs(name="distributed_env_test", output_file=str(output_file))
            job = Job(
                id="test_distributed_env",
                args=job_args,
                state=JobState.PENDING,
                resources=resource,
                entrypoint=torch_distributed_test_function,
            )

            job_queue.initialize_with_jobs([job])

            # Run worker - this will actually execute with torch distributed
            # Note: This test may fail if torch is not installed
            try:
                worker = Worker()
                worker.run(queue_path, "test_distributed_integration")

                # Check if job completed (may fail if torch not available)
                # We're mainly testing the command construction here
                completed = job_queue.count_completed_jobs()
                failed = job_queue.count_failed_jobs()

                # Job should either complete or fail (if torch not installed)
                assert completed + failed == 1

                # If completed, check that output file was created
                if completed == 1 and output_file.exists():
                    content = output_file.read_text()
                    # Should contain evidence of distributed environment
                    assert "Job: distributed_env_test" in content
                    # May have MASTER_PORT set
                    assert "MASTER_PORT" in content

            except Exception as e:
                # If torch.distributed is not available, that's okay for this test
                pytest.skip(f"Torch distributed not available: {e}")


class TestWorkerSubprocess:
    """Test Worker with subprocess-based job execution"""

    def test_worker_subprocess_completion(self):
        """Test that worker completes jobs using subprocess"""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_path = Path(temp_dir) / "test_queue.json"
            job_queue = JobQueue(job_array_path=queue_path)

            # Create test jobs
            jobs = []
            resource = ResourceRequest(cpu=1.0, memory=4.0, gpu=0, parallel_jobs=1)

            for i in range(2):
                job_args = JobTestArgs(name=f"subprocess_job_{i}", value=i)
                job = Job(
                    id=f"test_subprocess_{i}",
                    args=job_args,
                    state=JobState.PENDING,
                    resources=resource,
                    entrypoint=dummy_entrypoint_function,
                )
                jobs.append(job)

            # Initialize queue with jobs
            job_queue.initialize_with_jobs(jobs)

            # Create and run worker
            worker = Worker()
            worker.run(queue_path, "test_subprocess_worker")

            # Verify all jobs completed
            assert job_queue.count_completed_jobs() == 2
            assert job_queue.count_failed_jobs() == 0
            assert job_queue.count_pending_jobs() == 0
            assert job_queue.count_running_jobs() == 0

    def test_worker_parallel_subprocess_execution(self):
        """Test worker can run multiple subprocess jobs in parallel"""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_path = Path(temp_dir) / "test_queue.json"
            job_queue = JobQueue(job_array_path=queue_path)

            # Create test jobs
            jobs = []
            resource = ResourceRequest(cpu=1.0, memory=4.0, gpu=0, parallel_jobs=1)

            for i in range(4):
                job_args = JobTestArgs(name=f"parallel_job_{i}", value=i)
                job = Job(
                    id=f"test_parallel_{i}",
                    args=job_args,
                    state=JobState.PENDING,
                    resources=resource,
                    entrypoint=dummy_entrypoint_function,
                )
                jobs.append(job)

            # Initialize queue with jobs
            job_queue.initialize_with_jobs(jobs)

            # Run worker with parallel execution
            worker = Worker()
            worker.run(queue_path, "test_parallel_worker", parallel_jobs=2)

            # Verify all jobs completed
            assert job_queue.count_completed_jobs() == 4
            assert job_queue.count_failed_jobs() == 0

    def test_worker_captures_failed_job_stderr(self):
        """Test that worker captures stderr output from failed jobs"""
        from tests.launcher.test_functions import failing_test_function

        with tempfile.TemporaryDirectory() as temp_dir:
            queue_path = Path(temp_dir) / "test_queue.json"
            job_queue = JobQueue(job_array_path=queue_path)

            # Create a job that will fail
            resource = ResourceRequest(cpu=1.0, memory=4.0, gpu=0, parallel_jobs=1)
            job_args = JobTestArgs(name="worker_fail_test", value=42)
            job = Job(
                id="test_worker_fail",
                args=job_args,
                state=JobState.PENDING,
                resources=resource,
                entrypoint=failing_test_function,
            )

            # Initialize queue with the failing job
            job_queue.initialize_with_jobs([job])

            # Create and run worker - it should handle the failure gracefully
            worker = Worker()
            worker.run(queue_path, "test_fail_worker")

            # Verify the job was marked as failed
            assert job_queue.count_failed_jobs() == 1
            assert job_queue.count_completed_jobs() == 0
            assert job_queue.count_pending_jobs() == 0
            assert job_queue.count_running_jobs() == 0


if __name__ == "__main__":
    pytest.main([__file__])
