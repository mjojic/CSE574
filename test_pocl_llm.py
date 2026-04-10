#!/usr/bin/env python3
"""
Run PartialOrderPlanner with LLM-guided open-condition / action selection.

Requires OPENAI_API_KEY (e.g. in a .env file in this directory). Optional:
OPENAI_MODEL (defaults to gpt-5 in planning.py if unset in env).

Example:
  source ~/envs/meta/bin/activate
  python test_pocl_llm.py
"""

import os

from planning import PartialOrderPlanner, socks_and_shoes


def main():
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key, "
            "or export OPENAI_API_KEY in the shell."
        )

    problem = socks_and_shoes()
    model = os.getenv("OPENAI_MODEL")  # None -> planning uses env default in _init_llm
    planner = PartialOrderPlanner(
        problem,
        oc_heuristic="llm",
        openai_model=model,
    )
    planner.execute(display=True)


if __name__ == "__main__":
    main()
