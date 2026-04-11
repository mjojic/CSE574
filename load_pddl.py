"""Load PDDL domain + instance files into PlanningProblem / Action objects."""

import os

from planning import PlanningProblem, Action


# ── S-expression tokeniser / parser ─────────────────────────────────

def _tokenize(text):
    tokens, i = [], 0
    while i < len(text):
        c = text[i]
        if c in '()':
            tokens.append(c)
            i += 1
        elif c.isspace():
            i += 1
        elif c == ';':
            while i < len(text) and text[i] != '\n':
                i += 1
        else:
            j = i
            while j < len(text) and text[j] not in '() \t\n\r':
                j += 1
            tokens.append(text[i:j])
            i = j
    return tokens


def _parse_sexp(tokens):
    idx = [0]

    def _p():
        if tokens[idx[0]] == '(':
            idx[0] += 1
            lst = []
            while tokens[idx[0]] != ')':
                lst.append(_p())
            idx[0] += 1
            return lst
        tok = tokens[idx[0]]
        idx[0] += 1
        return tok

    return _p()


# ── PDDL → Expr-string helpers ─────────────────────────────────────

def _camel(name):
    """'pick-up' → 'PickUp', 'ontable' → 'Ontable'."""
    return ''.join(w.capitalize() for w in name.split('-'))


def _atom_str(sexp, obj_map=None):
    """['on', 'a', 'b'] → 'On(A, B)';  ['handempty'] → 'Handempty'."""
    if isinstance(sexp, str):
        return _camel(sexp)
    pred = _camel(sexp[0])
    args = []
    for a in sexp[1:]:
        if a.startswith('?'):
            args.append(a[1:])
        elif obj_map and a in obj_map:
            args.append(obj_map[a])
        else:
            args.append(a.upper())
    return f"{pred}({', '.join(args)})" if args else pred


def _formula_str(sexp, obj_map=None):
    """Parse a PDDL formula (handles 'and' / 'not') into an &-separated expr string."""
    items = sexp[1:] if isinstance(sexp, list) and sexp[0] == 'and' else [sexp]
    parts = []
    for item in items:
        if isinstance(item, list) and item[0] == 'not':
            parts.append('~' + _atom_str(item[1], obj_map))
        else:
            parts.append(_atom_str(item, obj_map))
    return ' & '.join(parts)


# ── Public API ──────────────────────────────────────────────────────

def parse_domain(filepath):
    """Parse a PDDL domain file and return a list of Action objects."""
    with open(filepath) as f:
        tree = _parse_sexp(_tokenize(f.read()))

    actions = []
    for section in tree:
        if not isinstance(section, list) or section[0] != ':action':
            continue
        name = section[1]
        kw = {}
        i = 2
        while i < len(section):
            if isinstance(section[i], str) and section[i].startswith(':'):
                kw[section[i]] = section[i + 1]
                i += 2
            else:
                i += 1

        params = [p[1:] for p in kw.get(':parameters', [])
                  if isinstance(p, str) and p.startswith('?')]
        act_name = _camel(name)
        act_str = f"{act_name}({', '.join(params)})" if params else act_name
        precond = _formula_str(kw[':precondition']) if ':precondition' in kw else ''
        effect = _formula_str(kw[':effect']) if ':effect' in kw else ''
        actions.append(Action(act_str, precond, effect))

    return actions


def parse_instance(filepath):
    """Parse a PDDL instance file.  Returns (objects, initial_str, goals_str)."""
    with open(filepath) as f:
        tree = _parse_sexp(_tokenize(f.read()))

    objects, init_atoms, goal_atoms = [], [], []
    for section in tree:
        if not isinstance(section, list):
            continue
        tag = section[0]
        if tag == ':objects':
            objects = [s for s in section[1:] if isinstance(s, str)]
        elif tag == ':init':
            init_atoms = section[1:]
        elif tag == ':goal':
            body = section[1]
            goal_atoms = body[1:] if isinstance(body, list) and body[0] == 'and' else [body]

    obj_map = {o: o.upper() for o in objects}
    init_str = ' & '.join(_atom_str(a, obj_map) for a in init_atoms)
    goal_str = ' & '.join(_atom_str(a, obj_map) for a in goal_atoms)
    return objects, init_str, goal_str


def load_problem(domain_path, instance_path):
    """Build a single PlanningProblem from a domain + instance file pair."""
    actions = parse_domain(domain_path)
    _, initial, goals = parse_instance(instance_path)
    return PlanningProblem(initial=initial, goals=goals, actions=actions)


def load_directory(domain_path, instance_dir):
    """Load every instance-*.pddl in *instance_dir*.

    Returns {filename: PlanningProblem} sorted by filename.
    """
    actions = parse_domain(domain_path)
    problems = {}
    for fname in sorted(os.listdir(instance_dir)):
        if not fname.endswith('.pddl') or not fname.startswith('instance'):
            continue
        _, initial, goals = parse_instance(os.path.join(instance_dir, fname))
        problems[fname] = PlanningProblem(
            initial=initial, goals=goals, actions=list(actions),
        )
    return problems


# ── Quick smoke-test ────────────────────────────────────────────────

if __name__ == '__main__':
    from utils import expr

    base = os.path.join(os.path.dirname(__file__), 'instances', 'blocksworld')
    domain = os.path.join(base, 'generated_domain.pddl')

    # --- single instance ---
    p = load_problem(domain, os.path.join(base, 'generated', 'instance-14.pddl'))
    print("=== instance-14  (3 blocks: L E G, all on table) ===")
    print("Initial:", p.initial)
    print("Goals:  ", p.goals)
    print("Actions:", p.actions)
    print("Goal met?", p.goal_test())

    p.act(expr('PickUp(E)'))
    p.act(expr('Stack(E, G)'))
    p.act(expr('PickUp(L)'))
    p.act(expr('Stack(L, E)'))
    print("After solving → goal met?", p.goal_test())

    # --- whole directory ---
    problems = load_directory(domain, os.path.join(base, 'generated'))
    print(f"\nLoaded {len(problems)} instances from generated/")

    # --- mystery domain works too ---
    mystery_domain = os.path.join(base, 'mystery', 'generated_domain.pddl')
    mystery = load_directory(mystery_domain, os.path.join(base, 'mystery', 'generated'))
    print(f"Loaded {len(mystery)} instances from mystery/generated/")
