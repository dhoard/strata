#!/usr/bin/env python3
"""The Ornith quality/regression prompt suite (ORNITH_QWEN35MOE.md phase 23, plus 18F).

Kept apart from tools/ornith_quality.py so the suite can be read, reviewed and unit-tested without
importing the engine, the server or jinja.  `category` is the workload phase 23 names; `prompt` is
the user turn, which ornith_quality.py renders through the model's own chat template.
"""
from __future__ import annotations

CASES: list[dict] = [
    dict(name="code-generate", category="code generation", prompt=(
        "Write a C++17 function `bool same_tree(const Node* a, const Node* b)` that returns true "
        "when two binary trees hold equal values in the same shape. Include the struct and the "
        "function. Then write one unit test for it.")),
    dict(name="code-repair", category="code repair", prompt=(
        "Review this implementation and explain how to fix the race. Then give the corrected "
        "`run()` method.\n"
        "class WorkQueue {\n"
        "    std::mutex mutex;\n"
        "    std::condition_variable ready;\n"
        "    std::deque<std::function<void()>> pending;\n"
        "    bool stopping = false;\n"
        "public:\n"
        "    void submit(std::function<void()> job) {\n"
        "        std::unique_lock lock(mutex);\n"
        "        pending.push_back(std::move(job));\n"
        "        ready.notify_one();\n"
        "    }\n"
        "    void run() {\n"
        "        while (!stopping) {\n"
        "            std::unique_lock lock(mutex);\n"
        "            ready.wait(lock, [&] { return stopping || !pending.empty(); });\n"
        "            if (stopping && pending.empty()) return;\n"
        "            auto job = std::move(pending.front());\n"
        "            pending.pop_front();\n"
        "            lock.unlock();\n"
        "            job();\n"
        "        }\n"
        "    }\n"
        "};")),
    dict(name="repo-reasoning", category="repository reasoning", prompt=(
        "A C++ project has these files: `src/core/session.cpp` (owns the per-request state), "
        "`src/kernels/attention.cu` (the GPU kernel), `src/core/session.hpp` (the interface), and "
        "`tests/session_test.cpp`. A bug report says the second request in a process returns the "
        "first request's tokens. Walk through where you would look, in order, and what you would "
        "check at each step before changing any code.")),
    dict(name="json-structured", category="structured JSON", prompt=(
        "Return only a JSON object describing a 40-layer hybrid language model. Use exactly these "
        "keys: architecture (string), layers (integer), recurrent_layers (integer), "
        "attention_layers (integer), experts (integer), experts_per_token (integer). No prose, no "
        "markdown fence.")),
    dict(name="prose-reasoning", category="prose reasoning", prompt=(
        "Explain, in plain language and without equations, why a mixture-of-experts model with a "
        "small number of selected experts per token can still be more accurate than a dense model "
        "of the same total parameter count. Give one concrete example, then one counter-example "
        "where it would not help.")),
    dict(name="instruction-following", category="instruction following", prompt=(
        "Answer with exactly three bullet points. Each bullet must begin with the word `Because`. "
        "Do not add an introduction, a conclusion, or any text outside the three bullets. The "
        "subject is why power supplies are rated in watts.")),
    dict(name="long-repeated", category="long repeated structure", prompt=(
        "Continue the sequence exactly, keeping the same pattern, one item per line, and add "
        "twelve more lines:\n"
        "item-001: alpha\nitem-002: beta\nitem-003: gamma")),
]
