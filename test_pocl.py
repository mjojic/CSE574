#!/usr/bin/env python3
"""Run PartialOrderPlanner on a bundled planning problem (smoke / demo)."""

from planning import PartialOrderPlanner, socks_and_shoes
from load_pddl import load_problem


def main():
    # problem = socks_and_shoes()
    problem = load_problem(domain_path="instances/blocksworld/generated_domain.pddl", instance_path="instances/blocksworld/generated/instance-3.pddl")
    print(problem.initial)
    print(problem.goals)
    print(problem.actions)
    for action in problem.actions:
        print(action.name)
        print(action.precond)
        print(action.effect)
        print(action.args)
        print(action.domain)
    planner = PartialOrderPlanner(problem)
    planner.execute(display=True)


if __name__ == "__main__":
    main()
