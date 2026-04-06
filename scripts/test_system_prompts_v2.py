"""Round 2: Iterate on top winners from round 1.
Focus on: json_aware (13) and structured_v1 (12).
Add web search emphasis and more test cases.
"""

import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

BASE_URL = "http://0.0.0.0:11556/v1"
API_KEY = "sk-xxx"
MODEL = "qwen3_5-27.36b"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ReadFile",
            "description": "Read a text file from disk.",
            "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "DuckDuckGoSearch",
            "description": "Search the web using DuckDuckGo.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ExecuteShell",
            "description": "Execute a shell command.",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "GlobTool",
            "description": "Find files matching a glob pattern.",
            "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "GrepTool",
            "description": "Search file contents with regex.",
            "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "FileEditTool",
            "description": "Replace exact text in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "WriteFile",
            "description": "Write content to a file.",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["file_path", "content"],
            },
        },
    },
]

TEST_CASES = [
    {"name": "greeting", "prompt": "hi, how are you?", "expect_tool": False, "expect_text": True},
    {"name": "read_file", "prompt": "read the file /etc/hostname", "expect_tool": True, "expect_tool_name": "ReadFile"},
    {
        "name": "web_search",
        "prompt": "search the web for Python 3.13 release date",
        "expect_tool": True,
        "expect_tool_name": "DuckDuckGoSearch",
    },
    {
        "name": "web_search2",
        "prompt": "look up the latest news about Rust programming language",
        "expect_tool": True,
        "expect_tool_name": "DuckDuckGoSearch",
    },
    {"name": "math", "prompt": "what is 2+2?", "expect_tool": False, "expect_text": True},
    {
        "name": "shell_cmd",
        "prompt": "run ls -la in the current directory",
        "expect_tool": True,
        "expect_tool_name": "ExecuteShell",
    },
    {
        "name": "find_files",
        "prompt": "find all python files in the current directory",
        "expect_tool": True,
        "expect_tool_name": "GlobTool",
    },
    {
        "name": "code_question",
        "prompt": "explain what a decorator is in python",
        "expect_tool": False,
        "expect_text": True,
    },
]

SYSTEM_PROMPTS = {
    "json_aware_v1": (
        "You are Calute, an AI coding assistant.\n\n"
        "You can call tools using the JSON function calling interface.\n"
        "Available tools: ReadFile, DuckDuckGoSearch, ExecuteShell, GlobTool, GrepTool, FileEditTool, WriteFile.\n\n"
        "Rules:\n"
        "1. Simple questions → answer directly, no tools.\n"
        "2. File operations → call ReadFile with file_path argument.\n"
        "3. Web search → call DuckDuckGoSearch with query argument.\n"
        "4. Shell commands → call ExecuteShell with command argument.\n"
        "5. Find files → call GlobTool with pattern argument.\n"
        "6. Search code → call GrepTool with pattern argument.\n"
        "7. NEVER output <tool_call> XML tags in your text. Use function calling only.\n"
        "8. Be concise.\n"
    ),
    "json_aware_v2_web_emphasis": (
        "You are Calute, an AI coding assistant.\n\n"
        "You can call tools using the JSON function calling interface.\n"
        "Available tools: ReadFile, DuckDuckGoSearch, ExecuteShell, GlobTool, GrepTool, FileEditTool, WriteFile.\n\n"
        "Rules:\n"
        "1. Simple questions → answer directly, no tools.\n"
        "2. File operations → call ReadFile.\n"
        "3. ANY web search, lookup, or internet query → call DuckDuckGoSearch. This includes 'search for', 'look up', 'find online', 'latest news about'.\n"
        "4. Shell commands → call ExecuteShell.\n"
        "5. Find files → call GlobTool.\n"
        "6. Search code → call GrepTool.\n"
        "7. NEVER output <tool_call> XML tags in your text. Use the function calling API.\n"
        "8. Be concise.\n"
    ),
    "structured_v2": (
        "You are Calute, an AI assistant running in a terminal.\n\n"
        "You have tools available via the JSON function calling interface. When you need to:\n"
        "- Read a file → call ReadFile(file_path=...)\n"
        "- Search the web → call DuckDuckGoSearch(query=...)\n"
        "- Run a command → call ExecuteShell(command=...)\n"
        "- Find files by pattern → call GlobTool(pattern=...)\n"
        "- Search code by regex → call GrepTool(pattern=...)\n"
        "- Edit a file → call FileEditTool(file_path=..., old_string=..., new_string=...)\n"
        "- Write a file → call WriteFile(file_path=..., content=...)\n\n"
        "Important:\n"
        "- Use the JSON function calling mechanism, NOT XML text-based tool tags.\n"
        "- For simple conversational messages, just respond in plain text.\n"
        "- When the user says 'search', 'look up', 'find online' → always use DuckDuckGoSearch.\n"
        "- Be concise and direct.\n"
    ),
    "structured_v3_concise": (
        "You are Calute, a terminal AI coding assistant with tool access.\n\n"
        "Tools (call via JSON function calling — NEVER via XML text):\n"
        "• ReadFile(file_path) — read a file\n"
        "• WriteFile(file_path, content) — write a file\n"
        "• FileEditTool(file_path, old_string, new_string) — edit a file\n"
        "• ExecuteShell(command) — run shell command\n"
        "• GlobTool(pattern) — find files by pattern\n"
        "• GrepTool(pattern) — search code by regex\n"
        "• DuckDuckGoSearch(query) — search the web\n\n"
        "Routing:\n"
        "• Simple question → answer directly\n"
        "• Needs file/web/shell → call the tool\n"
        "• 'search'/'look up'/'find online' → DuckDuckGoSearch\n"
    ),
    "numbered_rules_v1": (
        "You are Calute, an AI coding assistant with access to tools via function calling.\n\n"
        "# Tools\n"
        "ReadFile, WriteFile, FileEditTool, ExecuteShell, GlobTool, GrepTool, DuckDuckGoSearch\n\n"
        "# How to decide\n"
        "1. Can you answer from knowledge alone? → Reply directly.\n"
        "2. Need to read a file? → ReadFile(file_path=...)\n"
        "3. Need web info? → DuckDuckGoSearch(query=...)\n"
        "4. Need to run a command? → ExecuteShell(command=...)\n"
        "5. Need to find files? → GlobTool(pattern=...)\n"
        "6. Need to search code? → GrepTool(pattern=...)\n\n"
        "# Critical\n"
        "- Call tools via the function calling API. NEVER write <tool_call> XML in your text.\n"
        "- Be concise.\n"
    ),
    "minimal_json_v1": (
        "You are Calute, a coding assistant.\n"
        "Call tools via JSON function calling. Never write XML tool tags.\n"
        "Tools: ReadFile, WriteFile, FileEditTool, ExecuteShell, GlobTool, GrepTool, DuckDuckGoSearch.\n"
        "Answer simple questions directly. Use tools for file/web/shell tasks.\n"
    ),
    "explicit_routing_v1": (
        "You are Calute. You have 7 tools available through function calling:\n\n"
        "| Trigger | Tool | Argument |\n"
        "|---------|------|----------|\n"
        "| read/view a file | ReadFile | file_path |\n"
        "| write/create a file | WriteFile | file_path, content |\n"
        "| edit/replace in a file | FileEditTool | file_path, old_string, new_string |\n"
        "| run a command | ExecuteShell | command |\n"
        "| find files by pattern | GlobTool | pattern |\n"
        "| search code by regex | GrepTool | pattern |\n"
        "| search the web/internet | DuckDuckGoSearch | query |\n\n"
        "If no tool is needed, answer directly. Never write XML tool tags in text.\n"
    ),
    "hybrid_best_v1": (
        "You are Calute, an AI coding assistant.\n\n"
        "You can call tools using the JSON function calling interface.\n"
        "Available tools: ReadFile, WriteFile, FileEditTool, ExecuteShell, GlobTool, GrepTool, DuckDuckGoSearch.\n\n"
        "Rules:\n"
        "1. Simple questions (greetings, math, explanations) → answer directly, no tools.\n"
        "2. File read → ReadFile(file_path=...)\n"
        "3. File write → WriteFile(file_path=..., content=...)\n"
        "4. File edit → FileEditTool(file_path=..., old_string=..., new_string=...)\n"
        "5. Shell/command → ExecuteShell(command=...)\n"
        "6. Find files → GlobTool(pattern=...)\n"
        "7. Search code → GrepTool(pattern=...)\n"
        "8. Web search / look up / find online / latest news → DuckDuckGoSearch(query=...)\n"
        "9. NEVER output <tool_call> XML in your text. Always use the function calling API.\n"
        "10. Be concise and direct.\n"
    ),
}


def run_test(system_prompt, test_case):
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": test_case["prompt"]}]
    try:
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto", max_tokens=1024, temperature=0.7
        )
    except Exception as e:
        return {"error": str(e), "score": 0, "details": [f"error({e})"]}

    choice = response.choices[0]
    content = choice.message.content or ""
    tool_calls = choice.message.tool_calls or []
    has_xml = bool(re.search(r"<tool_call|<function=|</tool_call>", content))

    score = 0
    details = []

    if has_xml:
        score -= 3
        details.append("XML_LEAK(-3)")
    else:
        score += 1
        details.append("clean(+1)")

    if test_case.get("expect_tool"):
        if tool_calls:
            score += 2
            details.append("tool(+2)")
            names = [tc.function.name for tc in tool_calls]
            if test_case.get("expect_tool_name") in names:
                score += 1
                details.append(f"correct({test_case['expect_tool_name']})(+1)")
            else:
                details.append(f"wrong({names})")
        else:
            score -= 1
            details.append("no_tool(-1)")
    else:
        if tool_calls:
            score -= 1
            details.append("unwanted_tool(-1)")
        else:
            score += 1
            details.append("no_tool_ok(+1)")

    if test_case.get("expect_text") and content and len(content.strip()) > 5:
        score += 1
        details.append("text(+1)")

    return {"score": score, "details": details, "has_xml": has_xml, "tools": [tc.function.name for tc in tool_calls]}


def main():
    print(f"Round 2: {len(SYSTEM_PROMPTS)} prompts × {len(TEST_CASES)} tests")
    print(f"Model: {MODEL} @ {BASE_URL}\n{'=' * 70}")

    results = {}
    for pname, prompt in SYSTEM_PROMPTS.items():
        print(f"\n{'─' * 70}\n{pname}\n{'─' * 70}")
        total = 0
        for test in TEST_CASES:
            print(f"  {test['name']:20s}", end=" ", flush=True)
            r = run_test(prompt, test)
            total += r["score"]
            icon = "✓" if r["score"] > 0 else "✗" if r["score"] < 0 else "~"
            print(f"{icon} {r['score']:+d}  {' '.join(r['details'])}")
            time.sleep(0.3)
        results[pname] = total
        print(f"  {'TOTAL':20s} = {total}")

    print(f"\n{'=' * 70}\nRANKING\n{'=' * 70}")
    for i, (n, s) in enumerate(sorted(results.items(), key=lambda x: -x[1]), 1):
        print(f"  {i}. {n:30s} {s:+d}")

    best = max(results, key=results.get)
    print(f"\n✓ Best: {best} ({results[best]:+d})")
    print(f"\n{'─' * 70}\n{SYSTEM_PROMPTS[best]}\n{'─' * 70}")

    with open("scripts/prompt_test_results_v2.json", "w") as f:
        json.dump(
            {"ranking": sorted(results.items(), key=lambda x: -x[1]), "best": best, "best_prompt": SYSTEM_PROMPTS[best]},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
