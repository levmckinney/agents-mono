"""
Crash Debugger for Torchrun

Automatically launches a remote PDB server on an open port when an unhandled
exception occurs, allowing you to connect and inspect the state.

Usage:
    1. As a wrapper for modules:
       python -m crash_debugger -m your_module [args...]

    2. With a custom timeout (default 10 minutes):
       python -m crash_debugger --timeout 300 -m your_module [args...]

    3. With torchrun:
       torchrun --nproc_per_node=2 -m crash_debugger -m your_module [args...]

    4. As an import in your script:
       import crash_debugger
       crash_debugger.install(timeout=600)  # 10 minute timeout

    5. As a context manager:
       with crash_debugger.debug_on_crash(timeout=600):
           # your code here

When a crash occurs, you'll see output like:
    [Rank 0] PDB server listening on 0.0.0.0:5678
    [Rank 0] Connect with: nc localhost 5678

Connect using nc, telnet, or socat for readline support:
    socat readline tcp:localhost:5678
"""

import os
import pdb
import runpy
import socket
import sys
import traceback
import types
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Optional

DEFAULT_TIMEOUT = 600  # 10 minutes


def find_open_port(start: int = 5678, end: int = 5778) -> int:
    """Find an available port in the given range."""
    for port in range(start, end):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No open ports found in range {start}-{end}")


def get_rank() -> Optional[int]:
    """Get the distributed rank if running under torchrun."""
    for var in ["RANK", "LOCAL_RANK", "OMPI_COMM_WORLD_RANK"]:
        if var in os.environ:
            return int(os.environ[var])
    return None


class RemotePdb(pdb.Pdb):
    """PDB subclass that operates over a socket connection."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5678,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((host, port))
        self.server_socket.listen(1)
        self.server_socket.settimeout(timeout)

        self.host = host
        self.port = port
        self.timeout = timeout

        rank_str = f"[Rank {get_rank()}] " if get_rank() is not None else ""
        timeout_min = timeout / 60

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"{rank_str}PDB server listening on {host}:{port}", file=sys.stderr)
        print(f"{rank_str}Connect with:", file=sys.stderr)
        print(f"{rank_str}  nc localhost {port}", file=sys.stderr)
        print(f"{rank_str}  telnet localhost {port}", file=sys.stderr)
        print(f"{rank_str}  socat readline tcp:localhost:{port}  (for readline)", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)
        print(
            f"{rank_str}Waiting for debugger connection (timeout: {timeout_min:.0f}m)...",
            file=sys.stderr,
        )

        try:
            self.client_socket, address = self.server_socket.accept()
        except socket.timeout:
            print(f"{rank_str}Timeout waiting for debugger, exiting.", file=sys.stderr)
            self.server_socket.close()
            sys.exit(1)

        print(f"{rank_str}Debugger connected from {address}", file=sys.stderr)

        # Create file-like objects for the socket
        self.socket_file_in = self.client_socket.makefile("r")
        self.socket_file_out = self.client_socket.makefile("w")

        super().__init__(stdin=self.socket_file_in, stdout=self.socket_file_out)
        self.prompt = "(Pdb) "

    def do_quit(self, arg: str) -> bool:
        """Clean up and quit."""
        self._cleanup()
        result = super().do_quit(arg)
        return result if result is not None else True

    do_q = do_exit = do_quit

    def _cleanup(self) -> None:
        """Close all socket resources."""
        try:
            self.socket_file_in.close()
            self.socket_file_out.close()
            self.client_socket.close()
            self.server_socket.close()
        except Exception:
            pass


def remote_post_mortem(
    tb: types.TracebackType | None,
    host: str = "0.0.0.0",
    port: Optional[int] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> None:
    """Start a remote PDB session for post-mortem debugging."""
    if port is None:
        rank = get_rank()
        base_port = 5678 + (rank or 0) * 100
        port = find_open_port(start=base_port, end=base_port + 100)

    debugger = RemotePdb(host=host, port=port, timeout=timeout)
    debugger.reset()
    debugger.interaction(None, tb)


def post_mortem_hook(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_tb: types.TracebackType | None,
    original_hook: Callable[[type[BaseException], BaseException, types.TracebackType | None], None] | None = None,
    host: str = "0.0.0.0",
    timeout: float = DEFAULT_TIMEOUT,
) -> None:
    """Exception hook that launches a remote PDB server for post-mortem debugging."""
    # Print the traceback first
    print("\n", file=sys.stderr)
    traceback.print_exception(exc_type, exc_value, exc_tb)

    rank_str = f"[Rank {get_rank()}] " if get_rank() is not None else ""
    print(f"\n{rank_str}Crash detected! Launching remote PDB server...", file=sys.stderr)

    try:
        remote_post_mortem(exc_tb, host=host, timeout=timeout)
    except Exception as e:
        print(f"{rank_str}Failed to launch debug server: {e}", file=sys.stderr)
        if original_hook:
            original_hook(exc_type, exc_value, exc_tb)


def install(host: str = "0.0.0.0", timeout: float = DEFAULT_TIMEOUT) -> None:
    """
    Install the crash debugger as the default exception hook.

    Call this at the start of your script to enable automatic debugging on crash.
    """
    original_hook = sys.excepthook
    sys.excepthook = lambda *args: post_mortem_hook(
        *args, original_hook=original_hook, host=host, timeout=timeout
    )


@contextmanager
def debug_on_crash(
    host: str = "0.0.0.0", timeout: float = DEFAULT_TIMEOUT
) -> Generator[None, None, None]:
    """
    Context manager that launches a remote PDB server if an exception occurs.

    Usage:
        with debug_on_crash():
            # your code that might crash
    """
    try:
        yield
    except Exception:
        exc_type, exc_value, exc_tb = sys.exc_info()
        assert exc_type is not None and exc_value is not None
        post_mortem_hook(exc_type, exc_value, exc_tb, host=host, timeout=timeout)
        raise


def main() -> None:
    """Run a module with the crash debugger enabled."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a module with crash debugging enabled.",
        usage="python -m crash_debugger [options] -m <module> [args...]",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        metavar="SECONDS",
        help=f"timeout in seconds waiting for debugger (default: {DEFAULT_TIMEOUT}s / 10m)",
    )
    parser.add_argument(
        "-m",
        dest="module",
        metavar="MODULE",
        required=True,
        help="run library module as a script (like python -m)",
    )
    parser.add_argument(
        "args",
        nargs="*",
        help="arguments passed to the module",
    )

    args = parser.parse_args()

    # Install the exception hook with timeout
    install(timeout=args.timeout)

    sys.argv = [args.module, *args.args]
    runpy.run_module(args.module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
