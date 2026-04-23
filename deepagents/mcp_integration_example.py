"""playground/deepagents/mcp_integration_example.py

Minimal runnable example demonstrating:
- Starting a Playwright MCP server as a restartable subprocess (via `npx @playwright/mcp@latest`).
- Connecting from Python using `langchain_mcp_adapters.client.MultiServerMCPClient`
  and `langchain_mcp_adapters.tools.load_mcp_tools`.
- Passing loaded MCP tools into `deepagents.create_deep_agent` and invoking
  a simple agent call (navigate to example.com and capture a result).

Module-level usage notes:
- Configuration via environment variables:
  - `DEEPAGENT_MODEL` : model identifier passed to `deepagents.create_deep_agent` (default: "gpt-4o-mini")
  - `MCP_COMMAND`     : command to start MCP server (default: "npx @playwright/mcp@latest")
  - `MCP_ARGS`        : extra args appended to `MCP_COMMAND` (default: "")
  - `MCP_BASE_URL`    : base URL the MCP server will listen on (default: "http://127.0.0.1:9222")
- This example uses asyncio and ensures the subprocess is terminated on exit.
- Minimal pip requirements (install before running):
    pip install deepagents langchain-mcp-adapters playwright

Run example:
    python3 playground/deepagents/mcp_integration_example.py

Notes:
- This example prefers explicit, defensive call patterns to handle multiple agent APIs.
- Adjust `MCP_BASE_URL` if your MCP server uses a different host/port.
"""

import asyncio
import os
import shlex
import signal
import sys
import time
from contextlib import suppress
from urllib.parse import urlparse

# Third-party imports used by the example. These must be installed in your environment.
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools
except Exception as e:
    MultiServerMCPClient = None  # type: ignore
    load_mcp_tools = None  # type: ignore

try:
    import deepagents
except Exception:
    deepagents = None  # type: ignore


DEFAULT_MCP_COMMAND = "npx @playwright/mcp@latest"
DEFAULT_MCP_BASE_URL = "http://127.0.0.1:9222"
DEFAULT_DEEPAGENT_MODEL = "gpt-4o-mini"


def parse_mcp_command() -> list:
    raw_cmd = os.environ.get("MCP_COMMAND", DEFAULT_MCP_COMMAND)
    raw_args = os.environ.get("MCP_ARGS", "")
    parts = shlex.split(raw_cmd)
    if raw_args:
        parts += shlex.split(raw_args)
    return parts


def get_mcp_base_url() -> str:
    return os.environ.get("MCP_BASE_URL", DEFAULT_MCP_BASE_URL)


def get_deepagent_model() -> str:
    return os.environ.get("DEEPAGENT_MODEL", DEFAULT_DEEPAGENT_MODEL)


def _host_port_from_url(url: str):
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 9222
    return host, port


class PlaywrightMCPProcess:
    """
    Async context manager that starts the MCP subprocess and ensures it is
    terminated on exit. It also streams stdout/stderr to the parent process.
    """

    def __init__(self, cmd: list, base_url: str = DEFAULT_MCP_BASE_URL, ready_timeout: float = 20.0):
        self.cmd = cmd
        self.base_url = base_url
        self.ready_timeout = ready_timeout
        self.proc = None
        self._stdout_task = None
        self._stopped = False

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.stop()

    async def start(self):
        if self.proc and self.proc.returncode is None:
            return
        print(f"Starting MCP subprocess: {' '.join(self.cmd)}")
        self.proc = await asyncio.create_subprocess_exec(
            *self.cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        # stream logs in background
        if self.proc.stdout:
            self._stdout_task = asyncio.create_task(self._stream_output(self.proc.stdout))
        # wait until the process appears to be listening
        await self._wait_for_ready()

    async def _stream_output(self, stream):
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                try:
                    print("[MCP]", line.decode().rstrip())
                except Exception:
                    print("[MCP] <binary-output>")
        except asyncio.CancelledError:
            pass

    async def _wait_for_ready(self):
        host, port = _host_port_from_url(self.base_url)
        start = time.time()
        while True:
            # Check for process exit
            if self.proc.returncode is not None:
                raise RuntimeError(f"MCP subprocess exited prematurely with code {self.proc.returncode}")
            # Attempt to open a TCP connection to the host/port
            try:
                reader, writer = await asyncio.open_connection(host, port)
                writer.close()
                with suppress(Exception):
                    await writer.wait_closed()
                print(f"MCP server seems ready at {self.base_url}")
                return
            except Exception:
                if time.time() - start > self.ready_timeout:
                    raise TimeoutError(f"Timed out waiting for MCP to become ready at {self.base_url}")
                await asyncio.sleep(0.2)

    async def stop(self):
        if self._stopped:
            return
        self._stopped = True
        if self.proc:
            if self.proc.returncode is None:
                print("Terminating MCP subprocess...")
                try:
                    self.proc.terminate()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(self.proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    print("MCP subprocess did not terminate, killing...")
                    with suppress(Exception):
                        self.proc.kill()
                        await self.proc.wait()
            else:
                # already exited
                pass
        if self._stdout_task:
            self._stdout_task.cancel()
            with suppress(Exception):
                await self._stdout_task


async def call_agent_maybe_async(agent, prompt: str):
    """
    Helper to call the agent in a robust way, supporting different agent APIs:
    - async `arun` / `acall`
    - sync or async `run`
    - callable agent objects
    """
    if agent is None:
        raise RuntimeError("Agent is None; ensure `deepagents` and dependencies are installed.")
    # common async names
    if hasattr(agent, "arun"):
        return await agent.arun(prompt)
    if hasattr(agent, "acall"):
        return await agent.acall(prompt)
    if hasattr(agent, "run"):
        res = agent.run(prompt)
        if asyncio.iscoroutine(res):
            return await res
        return res
    if callable(agent):
        res = agent(prompt)
        if asyncio.iscoroutine(res):
            return await res
        return res
    raise RuntimeError("Unrecognized agent API; cannot call it.")


async def main():
    if MultiServerMCPClient is None or load_mcp_tools is None or deepagents is None:
        print("Required packages not found. Install with:")
        print("    pip install deepagents langchain-mcp-adapters playwright")
        sys.exit(1)

    cmd = parse_mcp_command()
    base_url = get_mcp_base_url()
    model = get_deepagent_model()

    # Provide a graceful shutdown on SIGINT/SIGTERM
    stop_event = asyncio.Event()

    def _signal_handler():
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows/other environments may not support add_signal_handler
            pass

    # Start MCP subprocess and create client -> tools -> deep agent -> run call
    async with PlaywrightMCPProcess(cmd=cmd, base_url=base_url) as mcp_proc:
        try:
            # Connect client to the MCP server. MultiServerMCPClient typically accepts
            # a list of server URLs; adjust as necessary for your version.
            client = MultiServerMCPClient([base_url])
            print("Connected MultiServerMCPClient to", base_url)

            # Load the MCP tools (browser/navigation/screenshot wrappers)
            tools = load_mcp_tools(client=client)
            print("Loaded MCP tools:", list(tools) if hasattr(tools, "__iter__") else type(tools))

            # Create a deep agent with the tools available.
            agent = deepagents.create_deep_agent(model=model, tools=tools)
            print("Deep agent created with model:", model)

            # Example prompt: navigate to example.com and return page title & short excerpt
            prompt = (
                "Using the browser/navigation tools, open https://example.com, "
                "retrieve the page title and the first 200 characters of visible text, "
                "and return them as a short JSON object."
            )
            print("Calling agent...")
            result = await call_agent_maybe_async(agent, prompt)
            print("Agent result:")
            print(result)

            # Wait for a termination signal (or exit immediately if you prefer)
            if not stop_event.is_set():
                # Short sleep to keep MCP alive briefly after the call so logs can flush
                await asyncio.sleep(0.5)
        finally:
            # Context manager will ensure MCP subprocess is terminated.
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted, exiting.")
