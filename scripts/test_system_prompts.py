"""Test system prompts against qwen3_5-27.36b to find the best one.

Scores each prompt on:
1. Does the model use proper function calling (not XML in content)?
2. Does the model respond directly to simple questions?
3. Does the model use tools when asked to read/search?
4. Does the model show reasoning_content?

Usage:
    python scripts/test_system_prompts.py
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

# ── Tool schemas (minimal set) ─────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ReadFile",
            "description": "Read a text file from disk and return its content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file to read."},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "DuckDuckGoSearch",
            "description": "Search the web using DuckDuckGo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ExecuteShell",
            "description": "Execute a shell command and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run."},
                },
                "required": ["command"],
            },
        },
    },
]

# ── Test cases ─────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "name": "simple_greeting",
        "prompt": "hi, how are you?",
        "expect_tool": False,
        "expect_text": True,
    },
    {
        "name": "read_file",
        "prompt": "read the file /etc/hostname",
        "expect_tool": True,
        "expect_tool_name": "ReadFile",
        "expect_text": False,
    },
    {
        "name": "web_search",
        "prompt": "search the web for Python 3.13 release date",
        "expect_tool": True,
        "expect_tool_name": "DuckDuckGoSearch",
        "expect_text": False,
    },
    {
        "name": "direct_answer",
        "prompt": "what is 2+2?",
        "expect_tool": False,
        "expect_text": True,
    },
]

# ── System prompts to test ─────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "minimal": (
        "You are Calute, a helpful AI coding assistant. "
        "You have access to tools for file operations, web search, and shell commands. "
        "Use tools when needed. Answer directly when you can."
    ),
    "explicit_tools": (
        "You are Calute, an AI coding assistant with tool access.\n\n"
        "# Tools Available\n"
        "- ReadFile: Read file contents\n"
        "- DuckDuckGoSearch: Search the web\n"
        "- ExecuteShell: Run shell commands\n\n"
        "# Rules\n"
        "- For simple questions, answer directly without tools.\n"
        "- For file/web/shell tasks, use the appropriate tool.\n"
        "- Never output XML tool tags in your response.\n"
        "- Use the function calling interface to invoke tools.\n"
    ),
    "structured_v1": (
        "You are Calute, an AI assistant running in a terminal.\n\n"
        "You have tools available via function calling. When you need to:\n"
        "- Read a file → call ReadFile\n"
        "- Search the web → call DuckDuckGoSearch\n"
        "- Run a command → call ExecuteShell\n\n"
        "Important:\n"
        "- Use the function calling mechanism, NOT XML or text-based tool invocation.\n"
        "- For simple conversational messages, just respond in plain text.\n"
        "- Be concise and direct.\n"
    ),
    "claude_style": (
        "You are Calute, an interactive AI coding agent running in the user's terminal.\n"
        "You help with software engineering tasks: writing code, debugging, refactoring, and more.\n\n"
        "# Available Tools\n\n"
        "## File & Shell\n"
        "- **ReadFile**: Read file contents with line numbers\n"
        "- **ExecuteShell**: Execute shell commands\n\n"
        "## Web\n"
        "- **DuckDuckGoSearch**: Search the web via DuckDuckGo\n\n"
        "# Guidelines\n"
        "- Be concise and direct. Lead with the answer.\n"
        "- Use tools only when you need live data, file access, or web search.\n"
        "- For simple questions, answer directly without calling tools.\n"
        "- If the user asks to search the web, use DuckDuckGoSearch.\n"
        "- Do not simulate tool calls or output XML/tool markup in your text responses.\n"
        "- Call tools using the function calling interface provided by the API.\n"
    ),
    "strict_no_xml": (
        "You are Calute, a terminal AI assistant.\n\n"
        "IMPORTANT: You have tools available through the API's function calling feature.\n"
        "NEVER write <tool_call> or any XML tags in your text output.\n"
        "NEVER simulate tool calls in your text.\n\n"
        "To use a tool, use the function calling mechanism — not text output.\n\n"
        "Available tools:\n"
        "- ReadFile(file_path) — reads a file\n"
        "- DuckDuckGoSearch(query) — web search\n"
        "- ExecuteShell(command) — run a shell command\n\n"
        "For simple questions (greetings, math, etc.), just answer in plain text.\n"
        "For tasks that need file/web/shell access, call the appropriate tool function.\n"
    ),
    "system_role_v1": (
        "You are a helpful coding assistant called Calute.\n"
        "You run in a terminal and have access to tools through function calling.\n\n"
        "When the user asks you to read a file, search the web, or run a command, "
        "use the provided function calling tools. Do not output tool call XML or markup.\n\n"
        "When the user asks a simple question, answer directly.\n"
    ),
    "json_aware": (
        "You are Calute, an AI coding assistant.\n\n"
        "You can call tools using the JSON function calling interface.\n"
        "Available tools: ReadFile, DuckDuckGoSearch, ExecuteShell.\n\n"
        "Rules:\n"
        "1. Simple questions → answer directly, no tools.\n"
        "2. File operations → call ReadFile with file_path argument.\n"
        "3. Web search → call DuckDuckGoSearch with query argument.\n"
        "4. Shell commands → call ExecuteShell with command argument.\n"
        "5. NEVER output <tool_call> XML tags in your text. Use function calling only.\n"
        "6. Be concise.\n"
    ),
    "cot_style": (
        "You are Calute, an AI assistant with access to tools.\n\n"
        "Before responding, think about whether you need a tool:\n"
        "- If the user asks a simple question → answer directly.\n"
        "- If the user asks to read/write files → use ReadFile tool.\n"
        "- If the user asks to search the web → use DuckDuckGoSearch tool.\n"
        "- If the user asks to run a command → use ExecuteShell tool.\n\n"
        "Call tools through the function calling API. Do not write XML tool tags.\n"
        "Be concise and helpful.\n"
    ),
}


def run_test(system_prompt: str, test_case: dict) -> dict:
    """Run a single test case and return results."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": test_case["prompt"]},
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=1024,
            temperature=0.7,
        )
    except Exception as e:
        return {"error": str(e), "score": 0}

    choice = response.choices[0]
    content = choice.message.content or ""
    tool_calls = choice.message.tool_calls or []
    reasoning = getattr(choice.message, "reasoning_content", None)
    if reasoning is None:
        try:
            reasoning = (choice.message.model_extra or {}).get("reasoning_content")
        except Exception:
            pass

    # Score
    score = 0
    details = []

    # Check for XML tool tags in content (BAD)
    has_xml_tools = bool(re.search(r"<tool_call|<function=|</tool_call>", content))
    if has_xml_tools:
        score -= 3
        details.append("XML_TOOL_TAGS_IN_CONTENT(-3)")
    else:
        score += 1
        details.append("no_xml_leak(+1)")

    # Check tool usage
    if test_case["expect_tool"]:
        if tool_calls:
            score += 2
            details.append("used_tool(+2)")
            if test_case.get("expect_tool_name"):
                names = [tc.function.name for tc in tool_calls]
                if test_case["expect_tool_name"] in names:
                    score += 1
                    details.append(f"correct_tool({test_case['expect_tool_name']})(+1)")
                else:
                    details.append(f"wrong_tool({names})(0)")
        else:
            score -= 1
            details.append("expected_tool_but_none(-1)")
    else:
        if tool_calls:
            score -= 1
            details.append("unexpected_tool_call(-1)")
        else:
            score += 1
            details.append("no_tool_correct(+1)")

    # Check text response
    if test_case["expect_text"]:
        if content and len(content.strip()) > 5:
            score += 1
            details.append("has_text(+1)")
        else:
            details.append("no_text(0)")

    # Bonus for reasoning
    if reasoning:
        score += 1
        details.append("has_reasoning(+1)")

    return {
        "score": score,
        "details": details,
        "content_preview": content[:150],
        "tool_calls": [tc.function.name for tc in tool_calls] if tool_calls else [],
        "has_reasoning": bool(reasoning),
        "has_xml_leak": has_xml_tools,
        "content_len": len(content),
    }


def main():
    print(f"Testing {len(SYSTEM_PROMPTS)} system prompts × {len(TEST_CASES)} test cases")
    print(f"Model: {MODEL} @ {BASE_URL}")
    print("=" * 70)

    results = {}

    for prompt_name, system_prompt in SYSTEM_PROMPTS.items():
        print(f"\n{'─' * 70}")
        print(f"Prompt: {prompt_name}")
        print(f"{'─' * 70}")

        total_score = 0
        prompt_results = []

        for test in TEST_CASES:
            print(f"  Test: {test['name']}...", end=" ", flush=True)
            result = run_test(system_prompt, test)
            total_score += result["score"]
            prompt_results.append({"test": test["name"], **result})

            status = "✓" if result["score"] > 0 else "✗"
            print(f"{status} score={result['score']} {' '.join(result.get('details', []))}")

            if result.get("has_xml_leak"):
                print(f"    ⚠ XML leak: {result['content_preview'][:80]}")

            time.sleep(0.5)  # Rate limit

        results[prompt_name] = {
            "total_score": total_score,
            "results": prompt_results,
        }
        print(f"  TOTAL: {total_score}")

    # Summary
    print("\n" + "=" * 70)
    print("RANKING")
    print("=" * 70)
    ranked = sorted(results.items(), key=lambda x: x[1]["total_score"], reverse=True)
    for i, (name, data) in enumerate(ranked, 1):
        xml_leaks = sum(1 for r in data["results"] if r.get("has_xml_leak"))
        reasoning = sum(1 for r in data["results"] if r.get("has_reasoning"))
        print(f"  {i}. {name:25s} score={data['total_score']:3d}  xml_leaks={xml_leaks}  reasoning={reasoning}")

    # Save
    best_name = ranked[0][0]
    print(f"\n✓ Best prompt: {best_name} (score={ranked[0][1]['total_score']})")

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_test_results.json")
    with open(output_path, "w") as f:
        json.dump(
            {"ranking": [(n, d["total_score"]) for n, d in ranked], "results": results, "best": best_name}, f, indent=2
        )
    print(f"Results saved to {output_path}")

    # Print the winning prompt
    print(f"\n{'─' * 70}")
    print(f"WINNING SYSTEM PROMPT ({best_name}):")
    print(f"{'─' * 70}")
    print(SYSTEM_PROMPTS[best_name])


if __name__ == "__main__":
    main()
