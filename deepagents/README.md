# DeepAgents MCP Sandbox

This folder contains a small prototype to demonstrate how a DeepAgents agent can use an MCP tool layer.

## Setup

1. Activate the sandbox env:
   ```bash
   cd /home/ccw/Documents/code/rnd/flutter-app/claim-agent/playground/deepagents
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Set your LLM provider credentials as needed.
   For example, for OpenAI:
   ```bash
   export OPENAI_API_KEY="YOUR_KEY"
   ```

3. Run the prototype:
   ```bash
   python sandbox_mcp_deepagent.py
   ```

## What it does

- Starts an MCP session using `langchain_mcp_adapters.MultiServerMCPClient`
- Loads MCP tools with `load_mcp_tools`
- Creates a DeepAgents agent via `create_deep_agent(...)`
- Sends a prompt that asks the agent to use Playwright actions

## Notes

- The script uses `npx @executeautomation/playwright-mcp-server`, so Node.js / npm must be installed.
- The example model defaults to `gpt-4o-mini`. Override with `DEEPAGENT_MODEL`.
- If you want to use a different MCP server, adjust `mcp_server_config` in `sandbox_mcp_deepagent.py`.
