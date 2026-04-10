#!/usr/bin/env python3
"""Run PartialOrderPlanner on a bundled planning problem (smoke / demo)."""

from planning import PartialOrderPlanner, socks_and_shoes


def main():
    problem = socks_and_shoes()
    planner = PartialOrderPlanner(problem)
    planner.execute(display=True)


if __name__ == "__main__":
    main()
