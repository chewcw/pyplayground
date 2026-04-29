"""Agentic vectorless RAG with PageIndex and LangChain deepagents.

This version keeps the same demo flow as example_agents.py, but swaps the
OpenAI Agents SDK runtime for LangChain's deepagents framework.
"""

import json
from pathlib import Path

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain_core.tools import tool

from tools.pageindex import PageIndexClient
from tools.pageindex.utils import print_tree

load_dotenv()

_EXAMPLES_DIR = Path(__file__).parent
PDF_PATH = _EXAMPLES_DIR / "assets" / "pc_employee_handbook_2026.pdf"
WORKSPACE = _EXAMPLES_DIR / "workspace"

AGENT_SYSTEM_PROMPT = """
You are PageIndex, a document QA assistant.

Use tools in this order when possible:
1. Call get_document() first to confirm status and page/line count.
2. Call get_document_structure() to identify relevant page ranges.
3. Call get_page_content(pages="5-7") with tight ranges; never fetch the whole document.

Answer only from tool output. Be concise.
"""


def _to_langchain_model_spec(model: str) -> str:
    """Adapt the repo's model strings to LangChain's provider syntax."""
    if not model:
        return model
    if ":" in model:
        return model

    if model.startswith("litellm/"):
        model = model.removeprefix("litellm/")

    if "/" not in model:
        return model

    provider, model_name = model.split("/", 1)
    if provider == "google":
        provider = "google_genai"

    if provider in {"openai", "anthropic", "google_genai"}:
        return f"{provider}:{model_name}"

    return model


def query_agent(client: PageIndexClient, doc_id: str, prompt: str, verbose: bool = False) -> str:
    """Run a document QA agent using deepagents."""

    @tool
    def get_document() -> str:
        """Get document metadata: status, page count, name, and description."""
        return client.get_document(doc_id)

    @tool
    def get_document_structure() -> str:
        """Get the document's full tree structure (without text) to find relevant sections."""
        return client.get_document_structure(doc_id)

    @tool
    def get_page_content(pages: str) -> str:
        """
        Get the text content of specific pages or line numbers.
        Use tight ranges: e.g. '5-7' for pages 5 to 7, '3,8' for pages 3 and 8, '12' for page 12.
        For Markdown documents, use line numbers from the structure's line_num field.
        """
        return client.get_page_content(doc_id, pages)

    model_spec = _to_langchain_model_spec(client.retrieve_model)
    if verbose:
        print(f"[deepagents] model: {model_spec}")
        print("[deepagents] tools: get_document, get_document_structure, get_page_content")

    agent = create_deep_agent(
        model=model_spec,
        tools=[get_document, get_document_structure, get_page_content],
        system_prompt=AGENT_SYSTEM_PROMPT,
    )

    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    messages = result.get("messages", []) if isinstance(result, dict) else getattr(result, "messages", [])
    if not messages:
        return ""

    final_message = messages[-1]
    content = getattr(final_message, "content", final_message)
    return content if isinstance(content, str) else str(content)


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF file not found: {PDF_PATH}")

    client = PageIndexClient(workspace=WORKSPACE)

    print("=" * 60)
    print("Step 1: Index PDF and view tree structure")
    print("=" * 60)
    doc_id = next(
        (did for did, doc in client.documents.items() if doc.get("doc_name") == PDF_PATH.name),
        None,
    )
    if doc_id:
        print(f"\nLoaded cached doc_id: {doc_id}")
    else:
        doc_id = client.index(str(PDF_PATH))
        print(f"\nIndexed. doc_id: {doc_id}")
    print("\nTree Structure (top-level sections):")
    structure = json.loads(client.get_document_structure(doc_id))
    print_tree(structure)

    print("\n" + "=" * 60)
    print("Step 2: View document metadata")
    print("=" * 60)
    doc_metadata = client.get_document(doc_id)
    print(f"\n{doc_metadata}")

    print("\n" + "=" * 60)
    print("Step 3: Agent Query (deepagents tool-use)")
    print("=" * 60)
    question = "What should a new employee know from this handbook?"
    print(f"\nQuestion: '{question}'")
    answer = query_agent(client, doc_id, question, verbose=True)
    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()