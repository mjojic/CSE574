"""Planning (Chapters 10-11) — classical problems, simulation, and partial-order planning."""

import itertools
import json
import os
import re
from functools import reduce as _reduce

from logic import FolKB, conjuncts, unify_mm
from utils import Expr, expr, first


class PlanningProblem:
    """
    Planning Domain Definition Language (PlanningProblem) used to define a search problem.
    It stores states in a knowledge base consisting of first order logic statements.
    The conjunction of these logical statements completely defines a state.
    """

    def __init__(self, initial, goals, actions, domain=None):
        self.initial = self.convert(initial) if domain is None else self.convert(initial) + self.convert(domain)
        self.goals = self.convert(goals)
        self.actions = actions
        self.domain = domain

    def convert(self, clauses):
        """Converts strings into exprs"""
        if not isinstance(clauses, Expr):
            if len(clauses) > 0:
                clauses = expr(clauses)
            else:
                clauses = []
        try:
            clauses = conjuncts(clauses)
        except AttributeError:
            pass

        new_clauses = []
        for clause in clauses:
            if clause.op == '~':
                new_clauses.append(expr('Not' + str(clause.args[0])))
            else:
                new_clauses.append(clause)
        return new_clauses

    def expand_actions(self, name=None):
        """Generate all possible actions with variable bindings for precondition selection heuristic"""

        has_domains = all(action.domain for action in self.actions if action.precond)
        kb = None
        if has_domains:
            kb = FolKB(self.initial)
            for action in self.actions:
                if action.precond:
                    kb.tell(expr(str(action.domain) + ' ==> ' + str(action)))

        objects = set(arg for clause in self.initial for arg in clause.args)
        expansions = []
        action_list = []
        if name is not None:
            for action in self.actions:
                if str(action.name) == name:
                    action_list.append(action)
                    break
        else:
            action_list = self.actions

        for action in action_list:
            for permutation in itertools.permutations(objects, len(action.args)):
                bindings = unify_mm(Expr(action.name, *action.args), Expr(action.name, *permutation))
                if bindings is not None:
                    new_args = []
                    for arg in action.args:
                        if arg in bindings:
                            new_args.append(bindings[arg])
                        else:
                            new_args.append(arg)
                    new_expr = Expr(str(action.name), *new_args)
                    if (has_domains and kb.ask(new_expr) is not False) or (
                            has_domains and not action.precond) or not has_domains:
                        new_preconds = []
                        for precond in action.precond:
                            new_precond_args = []
                            for arg in precond.args:
                                if arg in bindings:
                                    new_precond_args.append(bindings[arg])
                                else:
                                    new_precond_args.append(arg)
                            new_precond = Expr(str(precond.op), *new_precond_args)
                            new_preconds.append(new_precond)
                        new_effects = []
                        for effect in action.effect:
                            new_effect_args = []
                            for arg in effect.args:
                                if arg in bindings:
                                    new_effect_args.append(bindings[arg])
                                else:
                                    new_effect_args.append(arg)
                            new_effect = Expr(str(effect.op), *new_effect_args)
                            new_effects.append(new_effect)
                        expansions.append(Action(new_expr, new_preconds, new_effects))

        return expansions

    def goal_test(self):
        """Checks if the goals have been reached"""
        return all(goal in self.initial for goal in self.goals)

    def act(self, action):
        """
        Performs the action given as argument.
        Note that action is an Expr like expr('Remove(Glass, Table)') or expr('Eat(Sandwich)')
        """
        action_name = action.op
        args = action.args
        list_action = first(a for a in self.actions if a.name == action_name)
        if list_action is None:
            raise Exception("Action '{}' not found".format(action_name))
        if not list_action.check_precond(self.initial, args):
            raise Exception("Action '{}' pre-conditions not satisfied".format(action))
        self.initial = list_action(self.initial, args).clauses


class Action:
    """
    Defines an action schema using preconditions and effects.
    Use this to describe actions in PlanningProblem.
    action is an Expr where variables are given as arguments(args).
    Precondition and effect are both lists with positive and negative literals.
    Negative preconditions and effects are defined by adding a 'Not' before the name of the clause
    Example:
    precond = [expr("Human(person)"), expr("Hungry(Person)"), expr("NotEaten(food)")]
    effect = [expr("Eaten(food)"), expr("Hungry(person)")]
    eat = Action(expr("Eat(person, food)"), precond, effect)
    """

    def __init__(self, action, precond, effect, domain=None):
        if isinstance(action, str):
            action = expr(action)
        self.name = action.op
        self.args = action.args
        self.precond = self.convert(precond) if domain is None else self.convert(precond) + self.convert(domain)
        self.effect = self.convert(effect)
        self.domain = domain

    def __call__(self, kb, args):
        return self.act(kb, args)

    def __repr__(self):
        return '{}'.format(Expr(self.name, *self.args))

    def convert(self, clauses):
        """Converts strings into Exprs"""
        if isinstance(clauses, Expr):
            clauses = conjuncts(clauses)
            for i in range(len(clauses)):
                if clauses[i].op == '~':
                    clauses[i] = expr('Not' + str(clauses[i].args[0]))

        elif isinstance(clauses, str):
            clauses = clauses.replace('~', 'Not')
            if len(clauses) > 0:
                clauses = expr(clauses)

            try:
                clauses = conjuncts(clauses)
            except AttributeError:
                pass

        return clauses

    def substitute(self, e, args):
        """Replaces variables in expression with their respective Propositional symbol"""

        new_args = list(e.args)
        for num, x in enumerate(e.args):
            for i, _ in enumerate(self.args):
                if self.args[i] == x:
                    new_args[num] = args[i]
        return Expr(e.op, *new_args)

    def check_precond(self, kb, args):
        """Checks if the precondition is satisfied in the current state"""

        if isinstance(kb, list):
            kb = FolKB(kb)
        for clause in self.precond:
            if self.substitute(clause, args) not in kb.clauses:
                return False
        return True

    def act(self, kb, args):
        """Executes the action on the state's knowledge base"""

        if isinstance(kb, list):
            kb = FolKB(kb)

        if not self.check_precond(kb, args):
            raise Exception('Action pre-conditions not satisfied')
        for clause in self.effect:
            kb.tell(self.substitute(clause, args))
            if clause.op[:3] == 'Not':
                new_clause = Expr(clause.op[3:], *clause.args)

                if kb.ask(self.substitute(new_clause, args)) is not False:
                    kb.retract(self.substitute(new_clause, args))
            else:
                new_clause = Expr('Not' + clause.op, *clause.args)

                if kb.ask(self.substitute(new_clause, args)) is not False:
                    kb.retract(self.substitute(new_clause, args))

        return kb


def goal_test(goals, state):
    """Generic goal testing helper function"""

    if isinstance(state, list):
        kb = FolKB(state)
    else:
        kb = state
    return all(kb.ask(q) is not False for q in goals)


def air_cargo():
    """
    [Figure 10.1] AIR-CARGO-PROBLEM

    An air-cargo shipment problem for delivering cargo to different locations,
    given the starting location and airplanes.

    Example:
    >>> from planning import *
    >>> ac = air_cargo()
    >>> ac.goal_test()
    False
    >>> ac.act(expr('Load(C2, P2, JFK)'))
    >>> ac.act(expr('Load(C1, P1, SFO)'))
    >>> ac.act(expr('Fly(P1, SFO, JFK)'))
    >>> ac.act(expr('Fly(P2, JFK, SFO)'))
    >>> ac.act(expr('Unload(C2, P2, SFO)'))
    >>> ac.goal_test()
    False
    >>> ac.act(expr('Unload(C1, P1, JFK)'))
    >>> ac.goal_test()
    True
    >>>
    """

    return PlanningProblem(initial='At(C1, SFO) & At(C2, JFK) & At(P1, SFO) & At(P2, JFK)',
                           goals='At(C1, JFK) & At(C2, SFO)',
                           actions=[Action('Load(c, p, a)',
                                           precond='At(c, a) & At(p, a)',
                                           effect='In(c, p) & ~At(c, a)',
                                           domain='Cargo(c) & Plane(p) & Airport(a)'),
                                    Action('Unload(c, p, a)',
                                           precond='In(c, p) & At(p, a)',
                                           effect='At(c, a) & ~In(c, p)',
                                           domain='Cargo(c) & Plane(p) & Airport(a)'),
                                    Action('Fly(p, f, to)',
                                           precond='At(p, f)',
                                           effect='At(p, to) & ~At(p, f)',
                                           domain='Plane(p) & Airport(f) & Airport(to)')],
                           domain='Cargo(C1) & Cargo(C2) & Plane(P1) & Plane(P2) & Airport(SFO) & Airport(JFK)')


def spare_tire():
    """
    [Figure 10.2] SPARE-TIRE-PROBLEM

    A problem involving changing the flat tire of a car
    with a spare tire from the trunk.

    Example:
    >>> from planning import *
    >>> st = spare_tire()
    >>> st.goal_test()
    False
    >>> st.act(expr('Remove(Spare, Trunk)'))
    >>> st.act(expr('Remove(Flat, Axle)'))
    >>> st.goal_test()
    False
    >>> st.act(expr('PutOn(Spare, Axle)'))
    >>> st.goal_test()
    True
    >>>
    """

    return PlanningProblem(initial='At(Flat, Axle) & At(Spare, Trunk)',
                           goals='At(Spare, Axle) & At(Flat, Ground)',
                           actions=[Action('Remove(obj, loc)',
                                           precond='At(obj, loc)',
                                           effect='At(obj, Ground) & ~At(obj, loc)',
                                           domain='Tire(obj)'),
                                    Action('PutOn(t, Axle)',
                                           precond='At(t, Ground) & ~At(Flat, Axle)',
                                           effect='At(t, Axle) & ~At(t, Ground)',
                                           domain='Tire(t)'),
                                    Action('LeaveOvernight',
                                           precond='',
                                           effect='~At(Spare, Ground) & ~At(Spare, Axle) & ~At(Spare, Trunk) & \
                                        ~At(Flat, Ground) & ~At(Flat, Axle) & ~At(Flat, Trunk)')],
                           domain='Tire(Flat) & Tire(Spare)')


def three_block_tower():
    """
    [Figure 10.3] THREE-BLOCK-TOWER

    A blocks-world problem of stacking three blocks in a certain configuration,
    also known as the Sussman Anomaly.

    Example:
    >>> from planning import *
    >>> tbt = three_block_tower()
    >>> tbt.goal_test()
    False
    >>> tbt.act(expr('MoveToTable(C, A)'))
    >>> tbt.act(expr('Move(B, Table, C)'))
    >>> tbt.goal_test()
    False
    >>> tbt.act(expr('Move(A, Table, B)'))
    >>> tbt.goal_test()
    True
    >>>
    """
    return PlanningProblem(initial='On(A, Table) & On(B, Table) & On(C, A) & Clear(B) & Clear(C)',
                           goals='On(A, B) & On(B, C)',
                           actions=[Action('Move(b, x, y)',
                                           precond='On(b, x) & Clear(b) & Clear(y)',
                                           effect='On(b, y) & Clear(x) & ~On(b, x) & ~Clear(y)',
                                           domain='Block(b) & Block(y)'),
                                    Action('MoveToTable(b, x)',
                                           precond='On(b, x) & Clear(b)',
                                           effect='On(b, Table) & Clear(x) & ~On(b, x)',
                                           domain='Block(b) & Block(x)')],
                           domain='Block(A) & Block(B) & Block(C)')


def simple_blocks_world():
    """
    SIMPLE-BLOCKS-WORLD

    A simplified definition of the Sussman Anomaly problem.

    Example:
    >>> from planning import *
    >>> sbw = simple_blocks_world()
    >>> sbw.goal_test()
    False
    >>> sbw.act(expr('ToTable(A, B)'))
    >>> sbw.act(expr('FromTable(B, A)'))
    >>> sbw.goal_test()
    False
    >>> sbw.act(expr('FromTable(C, B)'))
    >>> sbw.goal_test()
    True
    >>>
    """

    return PlanningProblem(initial='On(A, B) & Clear(A) & OnTable(B) & OnTable(C) & Clear(C)',
                           goals='On(B, A) & On(C, B)',
                           actions=[Action('ToTable(x, y)',
                                           precond='On(x, y) & Clear(x)',
                                           effect='~On(x, y) & Clear(y) & OnTable(x)'),
                                    Action('FromTable(y, x)',
                                           precond='OnTable(y) & Clear(y) & Clear(x)',
                                           effect='~OnTable(y) & ~Clear(x) & On(y, x)')])


def have_cake_and_eat_cake_too():
    """
    [Figure 10.7] CAKE-PROBLEM

    A problem where we begin with a cake and want to
    reach the state of having a cake and having eaten a cake.
    The possible actions include baking a cake and eating a cake.

    Example:
    >>> from planning import *
    >>> cp = have_cake_and_eat_cake_too()
    >>> cp.goal_test()
    False
    >>> cp.act(expr('Eat(Cake)'))
    >>> cp.goal_test()
    False
    >>> cp.act(expr('Bake(Cake)'))
    >>> cp.goal_test()
    True
    >>>
    """

    return PlanningProblem(initial='Have(Cake)',
                           goals='Have(Cake) & Eaten(Cake)',
                           actions=[Action('Eat(Cake)',
                                           precond='Have(Cake)',
                                           effect='Eaten(Cake) & ~Have(Cake)'),
                                    Action('Bake(Cake)',
                                           precond='~Have(Cake)',
                                           effect='Have(Cake)')])


def shopping_problem():
    """
    SHOPPING-PROBLEM

    A problem of acquiring some items given their availability at certain stores.

    Example:
    >>> from planning import *
    >>> sp = shopping_problem()
    >>> sp.goal_test()
    False
    >>> sp.act(expr('Go(Home, HW)'))
    >>> sp.act(expr('Buy(Drill, HW)'))
    >>> sp.act(expr('Go(HW, SM)'))
    >>> sp.act(expr('Buy(Banana, SM)'))
    >>> sp.goal_test()
    False
    >>> sp.act(expr('Buy(Milk, SM)'))
    >>> sp.goal_test()
    True
    >>>
    """

    return PlanningProblem(initial='At(Home) & Sells(SM, Milk) & Sells(SM, Banana) & Sells(HW, Drill)',
                           goals='Have(Milk) & Have(Banana) & Have(Drill)',
                           actions=[Action('Buy(x, store)',
                                           precond='At(store) & Sells(store, x)',
                                           effect='Have(x)',
                                           domain='Store(store) & Item(x)'),
                                    Action('Go(x, y)',
                                           precond='At(x)',
                                           effect='At(y) & ~At(x)',
                                           domain='Place(x) & Place(y)')],
                           domain='Place(Home) & Place(SM) & Place(HW) & Store(SM) & Store(HW) & '
                                  'Item(Milk) & Item(Banana) & Item(Drill)')


def socks_and_shoes():
    """
    SOCKS-AND-SHOES-PROBLEM

    A task of wearing socks and shoes on both feet

    Example:
    >>> from planning import *
    >>> ss = socks_and_shoes()
    >>> ss.goal_test()
    False
    >>> ss.act(expr('RightSock'))
    >>> ss.act(expr('RightShoe'))
    >>> ss.act(expr('LeftSock'))
    >>> ss.goal_test()
    False
    >>> ss.act(expr('LeftShoe'))
    >>> ss.goal_test()
    True
    >>>
    """

    return PlanningProblem(initial='',
                           goals='RightShoeOn & LeftShoeOn',
                           actions=[Action('RightShoe',
                                           precond='RightSockOn',
                                           effect='RightShoeOn'),
                                    Action('RightSock',
                                           precond='',
                                           effect='RightSockOn'),
                                    Action('LeftShoe',
                                           precond='LeftSockOn',
                                           effect='LeftShoeOn'),
                                    Action('LeftSock',
                                           precond='',
                                           effect='LeftSockOn')])


def double_tennis_problem():
    """
    [Figure 11.10] DOUBLE-TENNIS-PROBLEM

    A multiagent planning problem involving two partner tennis players
    trying to return an approaching ball and repositioning around in the court.

    Example:
    >>> from planning import *
    >>> dtp = double_tennis_problem()
    >>> goal_test(dtp.goals, dtp.initial)
    False
    >>> dtp.act(expr('Go(A, RightBaseLine, LeftBaseLine)'))
    >>> dtp.act(expr('Hit(A, Ball, RightBaseLine)'))
    >>> goal_test(dtp.goals, dtp.initial)
    False
    >>> dtp.act(expr('Go(A, LeftNet, RightBaseLine)'))
    >>> goal_test(dtp.goals, dtp.initial)
    True
    >>>
    """

    return PlanningProblem(
        initial='At(A, LeftBaseLine) & At(B, RightNet) & Approaching(Ball, RightBaseLine) & Partner(A, B) & Partner(B, A)',
        goals='Returned(Ball) & At(a, LeftNet) & At(a, RightNet)',
        actions=[Action('Hit(actor, Ball, loc)',
                        precond='Approaching(Ball, loc) & At(actor, loc)',
                        effect='Returned(Ball)'),
                 Action('Go(actor, to, loc)',
                        precond='At(actor, loc)',
                        effect='At(actor, to) & ~At(actor, loc)')])


class PartialOrderPlanner:
    """
    [Section 10.13] PARTIAL-ORDER-PLANNER

    Partially ordered plans are created by a search through the space of plans
    rather than a search through the state space. It views planning as a refinement of partially ordered plans.
    A partially ordered plan is defined by a set of actions and a set of constraints of the form A < B,
    which denotes that action A has to be performed before action B.
    To summarize the working of a partial order planner,
    1. An open precondition is selected (a sub-goal that we want to achieve).
    2. An action that fulfils the open precondition is chosen.
    3. Temporal constraints are updated.
    4. Existing causal links are protected. Protection is a method that checks if the causal links conflict
       and if they do, temporal constraints are added to fix the threats.
    5. The set of open preconditions is updated.
    6. Temporal constraints of the selected action and the next action are established.
    7. A new causal link is added between the selected action and the owner of the open precondition.
    8. The set of new causal links is checked for threats and if found, the threat is removed by either promotion or
       demotion. If promotion or demotion is unable to solve the problem, the planning problem cannot be solved with
       the current sequence of actions or it may not be solvable at all.
    9. These steps are repeated until the set of open preconditions is empty.
    """

    def __init__(self, planning_problem, oc_heuristic='default', openai_model=None):
        self.tries = 1
        self.planning_problem = planning_problem
        self.oc_heuristic = oc_heuristic
        self.causal_links = []
        self.start = Action('Start', [], self.planning_problem.initial)
        self.finish = Action('Finish', self.planning_problem.goals, [])
        self.actions = set()
        self.actions.add(self.start)
        self.actions.add(self.finish)
        self.constraints = set()
        self.constraints.add((self.start, self.finish))
        self.agenda = set()
        for precond in self.finish.precond:
            self.agenda.add((precond, self.finish))
        self.expanded_actions = planning_problem.expand_actions()

        self._openai_client = None
        self._openai_model = None
        if oc_heuristic == 'llm':
            self._init_llm(openai_model)

    def find_open_precondition(self):
        """Find open precondition with the least number of possible actions"""

        number_of_ways = dict()
        actions_for_precondition = dict()
        for element in self.agenda:
            open_precondition = element[0]
            possible_actions = list(self.actions) + self.expanded_actions
            for action in possible_actions:
                for effect in action.effect:
                    if effect == open_precondition:
                        if open_precondition in number_of_ways:
                            number_of_ways[open_precondition] += 1
                            actions_for_precondition[open_precondition].append(action)
                        else:
                            number_of_ways[open_precondition] = 1
                            actions_for_precondition[open_precondition] = [action]

        number = sorted(number_of_ways, key=number_of_ways.__getitem__)

        for k, v in number_of_ways.items():
            if v == 0:
                return None, None, None

        act1 = None
        for element in self.agenda:
            if element[0] == number[0]:
                act1 = element[1]
                break

        if number[0] in self.expanded_actions:
            self.expanded_actions.remove(number[0])

        return number[0], act1, actions_for_precondition[number[0]]

    # ---- LLM-guided open-condition / action selection ----

    def _init_llm(self, openai_model):
        """Initialize the OpenAI client for LLM-guided heuristic."""
        try:
            from dotenv import load_dotenv
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai and python-dotenv are required for oc_heuristic='llm'. "
                "Install with: pip install openai python-dotenv"
            )
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Create a .env file with your key "
                "or set the OPENAI_API_KEY environment variable."
            )
        self._openai_client = OpenAI(api_key=api_key)
        self._openai_model = openai_model or os.getenv('OPENAI_MODEL', 'gpt-5')

    def _openai_chat(self, messages):
        """Send a chat completion request and return the response text."""
        # Do not pass temperature: some models (e.g. gpt-5) only accept the API default.
        response = self._openai_client.chat.completions.create(
            model=self._openai_model,
            messages=messages,
        )
        return response.choices[0].message.content.strip()

    def _parse_json_response(self, text, key):
        """Extract an integer value for *key* from a JSON response string."""
        try:
            data = json.loads(text)
            return int(data[key])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            match = re.search(r'\{[^}]+\}', text)
            if match:
                try:
                    data = json.loads(match.group())
                    return int(data[key])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    pass
            raise ValueError(f"Could not parse '{key}' from LLM response: {text}")

    def _format_literals(self, literals):
        """Format a precondition or effect list for display."""
        if not literals:
            return '(none)'
        if isinstance(literals, list):
            return ', '.join(str(lit) for lit in literals)
        return str(literals)

    def _format_domain(self):
        """Return a human-readable description of the planning domain."""
        lines = []
        initial = self.planning_problem.initial
        lines.append("Initial state: " + (
            ', '.join(str(lit) for lit in initial) if initial else '(empty)'))
        lines.append("Goal: " + ', '.join(
            str(g) for g in self.planning_problem.goals))
        lines.append("\nAction Schemas:")
        for i, action in enumerate(self.planning_problem.actions, 1):
            header = action.name
            if action.args:
                header += "({})".format(', '.join(str(a) for a in action.args))
            lines.append("  {}. {}".format(i, header))
            lines.append("     Preconditions: {}".format(
                self._format_literals(action.precond)))
            lines.append("     Effects: {}".format(
                self._format_literals(action.effect)))
        return '\n'.join(lines)

    def _format_plan_state(self):
        """Return a human-readable snapshot of the current partial plan."""
        lines = ["Current Plan State:"]
        action_names = sorted(str(a) for a in self.actions)
        lines.append("  Actions in plan: {}".format(', '.join(action_names)))
        lines.append("  Ordering constraints:")
        for a, b in sorted(self.constraints,
                           key=lambda c: (str(c[0]), str(c[1]))):
            lines.append("    {} must occur before {}".format(a, b))
        if self.causal_links:
            lines.append("  Causal links:")
            for provider, literal, consumer in self.causal_links:
                lines.append(
                    "    {} achieves {} for the benefit of {}".format(
                        provider, literal, consumer))
        else:
            lines.append("  Causal links: (none yet)")
        return '\n'.join(lines)

    def _enumerate_open_conditions(self):
        """Build a stable list of (G, act1, possible_actions) from the agenda."""
        agenda_list = sorted(self.agenda,
                             key=lambda x: (str(x[0]), str(x[1])))
        candidates = []
        for G, act1 in agenda_list:
            possible_actions = []
            for action in list(self.actions) + self.expanded_actions:
                for effect in action.effect:
                    if effect == G:
                        possible_actions.append(action)
                        break
            if possible_actions:
                candidates.append((G, act1, possible_actions))
        return candidates

    def _select_open_condition_llm(self):
        """Use the LLM to choose which open condition to resolve next."""
        candidates = self._enumerate_open_conditions()
        if not candidates:
            raise ValueError("No open conditions have achievable actions")

        domain_desc = self._format_domain()
        plan_state = self._format_plan_state()

        oc_lines = []
        for i, (G, act1, possible) in enumerate(candidates):
            achievers = ', '.join(str(a) for a in possible)
            oc_lines.append(
                '  {}: "{}" is required by action "{}" '
                '(can be achieved by: {})'.format(i, G, act1, achievers))
        oc_text = '\n'.join(oc_lines)

        prompt = (
            "{}\n\n"
            "{}\n\n"
            "The following open conditions need to be resolved. Each one is a "
            "precondition of some action in the plan that is not yet supported "
            "by a causal link:\n"
            "{}\n\n"
            "Select the open condition that should be resolved next. Consider "
            "which choice would best advance the plan toward completion while "
            "minimizing ordering conflicts.\n\n"
            'Respond with ONLY a JSON object: {{"open_condition_index": <integer>}}'
        ).format(domain_desc, plan_state, oc_text)

        system_msg = (
            "You are an expert AI planner assisting a Partial-Order Causal Link "
            "(POCL) planning algorithm. You select open conditions and actions to "
            "resolve them. Always respond with valid JSON only, no extra text.")
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

        response_text = self._openai_chat(messages)
        try:
            idx = self._parse_json_response(response_text, 'open_condition_index')
        except ValueError:
            # Retry once with a corrective nudge
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content":
                "Your response was not valid JSON. Please respond with ONLY: "
                '{"open_condition_index": <integer>}'})
            response_text = self._openai_chat(messages)
            idx = self._parse_json_response(response_text, 'open_condition_index')

        if not 0 <= idx < len(candidates):
            raise ValueError(
                "LLM returned out-of-range index {} (valid: 0-{})".format(
                    idx, len(candidates) - 1))
        return candidates[idx]

    def _select_resolver_action_llm(self, G, act1, possible_actions):
        """Use the LLM to choose which action resolves the open condition."""
        if len(possible_actions) == 1:
            return possible_actions[0]

        domain_desc = self._format_domain()
        plan_state = self._format_plan_state()

        action_lines = []
        for i, action in enumerate(possible_actions):
            effects = self._format_literals(action.effect)
            preconds = self._format_literals(action.precond)
            action_lines.append(
                "  {}: {} (preconditions: {}; effects: {})".format(
                    i, action, preconds, effects))
        action_text = '\n'.join(action_lines)

        prompt = (
            "{}\n\n"
            "{}\n\n"
            'The open condition to resolve is: "{}" is required by action '
            '"{}".\n\n'
            "The following actions can achieve this condition:\n"
            "{}\n\n"
            "Select the best action to resolve this open condition. Consider "
            "precondition complexity, potential ordering conflicts with existing "
            "causal links, and overall plan efficiency.\n\n"
            'Respond with ONLY a JSON object: {{"action_index": <integer>}}'
        ).format(domain_desc, plan_state, G, act1, action_text)

        system_msg = (
            "You are an expert AI planner assisting a Partial-Order Causal Link "
            "(POCL) planning algorithm. You select actions to resolve open "
            "conditions. Always respond with valid JSON only, no extra text.")
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

        response_text = self._openai_chat(messages)
        try:
            idx = self._parse_json_response(response_text, 'action_index')
        except ValueError:
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content":
                "Your response was not valid JSON. Please respond with ONLY: "
                '{"action_index": <integer>}'})
            response_text = self._openai_chat(messages)
            idx = self._parse_json_response(response_text, 'action_index')

        if not 0 <= idx < len(possible_actions):
            raise ValueError(
                "LLM returned out-of-range index {} (valid: 0-{})".format(
                    idx, len(possible_actions) - 1))
        return possible_actions[idx]

    def cyclic(self, graph):
        """Check cyclicity of a directed graph"""

        new_graph = dict()
        for element in graph:
            if element[0] in new_graph:
                new_graph[element[0]].append(element[1])
            else:
                new_graph[element[0]] = [element[1]]

        path = set()

        def visit(vertex):
            path.add(vertex)
            for neighbor in new_graph.get(vertex, ()):
                if neighbor in path or visit(neighbor):
                    return True
            path.remove(vertex)
            return False

        value = any(visit(v) for v in new_graph)
        return value

    def add_const(self, constraint, constraints):
        """Add the constraint to constraints if the resulting graph is acyclic"""

        if constraint[0] == self.finish or constraint[1] == self.start:
            return constraints

        new_constraints = set(constraints)
        new_constraints.add(constraint)

        if self.cyclic(new_constraints):
            return constraints
        return new_constraints

    def is_a_threat(self, precondition, effect):
        """Check if effect is a threat to precondition"""

        if (str(effect.op) == 'Not' + str(precondition.op)) or ('Not' + str(effect.op) == str(precondition.op)):
            if effect.args == precondition.args:
                return True
        return False

    def protect(self, causal_link, action, constraints):
        """Check and resolve threats by promotion or demotion"""

        threat = False
        for effect in action.effect:
            if self.is_a_threat(causal_link[1], effect):
                threat = True
                break

        if action != causal_link[0] and action != causal_link[2] and threat:
            # try promotion
            new_constraints = set(constraints)
            new_constraints.add((action, causal_link[0]))
            if not self.cyclic(new_constraints):
                constraints = self.add_const((action, causal_link[0]), constraints)
            else:
                # try demotion
                new_constraints = set(constraints)
                new_constraints.add((causal_link[2], action))
                if not self.cyclic(new_constraints):
                    constraints = self.add_const((causal_link[2], action), constraints)
                else:
                    # both promotion and demotion fail
                    print('Unable to resolve a threat caused by', action, 'onto', causal_link)
                    return
        return constraints

    def convert(self, constraints):
        """Convert constraints into a dict of Action to set orderings"""

        graph = dict()
        for constraint in constraints:
            if constraint[0] in graph:
                graph[constraint[0]].add(constraint[1])
            else:
                graph[constraint[0]] = set()
                graph[constraint[0]].add(constraint[1])
        return graph

    def toposort(self, graph):
        """Generate topological ordering of constraints"""

        if len(graph) == 0:
            return

        graph = graph.copy()

        for k, v in graph.items():
            v.discard(k)

        extra_elements_in_dependencies = _reduce(set.union, graph.values()) - set(graph.keys())

        graph.update({element: set() for element in extra_elements_in_dependencies})
        while True:
            ordered = set(element for element, dependency in graph.items() if len(dependency) == 0)
            if not ordered:
                break
            yield ordered
            graph = {element: (dependency - ordered)
                     for element, dependency in graph.items()
                     if element not in ordered}
        if len(graph) != 0:
            raise ValueError('The graph is not acyclic and cannot be linearly ordered')

    def display_plan(self):
        """Display causal links, constraints and the plan"""

        print('Causal Links')
        for causal_link in self.causal_links:
            print(causal_link)

        print('\nConstraints')
        for constraint in self.constraints:
            print(constraint[0], '<', constraint[1])

        print('\nPartial Order Plan')
        print(list(reversed(list(self.toposort(self.convert(self.constraints))))))

    def execute(self, display=True):
        """Execute the algorithm"""

        step = 1
        while len(self.agenda) > 0:
            step += 1

            if self.oc_heuristic == 'llm':
                try:
                    G, act1, possible_actions = self._select_open_condition_llm()
                    act0 = self._select_resolver_action_llm(
                        G, act1, possible_actions)
                except (IndexError, ValueError) as e:
                    print('LLM selection failed: {}'.format(e))
                    break
            else:
                # select <G, act1> from Agenda
                try:
                    G, act1, possible_actions = self.find_open_precondition()
                except IndexError:
                    print('Probably Wrong')
                    break
                act0 = possible_actions[0]

            # remove <G, act1> from Agenda
            self.agenda.remove((G, act1))

            # Actions = Actions U {act0}
            self.actions.add(act0)

            # Constraints = add_const(start < act0, Constraints)
            self.constraints = self.add_const((self.start, act0), self.constraints)

            # for each CL E CausalLinks do
            #   Constraints = protect(CL, act0, Constraints)
            for causal_link in self.causal_links:
                self.constraints = self.protect(causal_link, act0, self.constraints)

            # Agenda = Agenda U {<P, act0>: P is a precondition of act0}
            for precondition in act0.precond:
                self.agenda.add((precondition, act0))

            # Constraints = add_const(act0 < act1, Constraints)
            self.constraints = self.add_const((act0, act1), self.constraints)

            # CausalLinks U {<act0, G, act1>}
            if (act0, G, act1) not in self.causal_links:
                self.causal_links.append((act0, G, act1))

            # for each A E Actions do
            #   Constraints = protect(<act0, G, act1>, A, Constraints)
            for action in self.actions:
                self.constraints = self.protect((act0, G, act1), action, self.constraints)

            if step > 200:
                print("Couldn't find a solution")
                return None, None

        if display:
            self.display_plan()
        else:
            return self.constraints, self.causal_links
