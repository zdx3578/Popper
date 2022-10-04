# import clingo
# import clingo.script
# import pkg_resources
# from . core import Literal, ConstVar
# from collections import defaultdict
# from . util import rule_is_recursive
# clingo.script.enable_python()

# arg_lookup = {clingo.Number(i):chr(ord('A') + i) for i in range(100)}

# # TODO: COULD CACHE TUPLES OF ARGS FOR TINY OPTIMISATION
# def parse_model(model):
#     directions = defaultdict(lambda: defaultdict(lambda: '?'))
#     rule_index_to_body = defaultdict(set)
#     rule_index_to_head = {}
#     rule_index_ordering = defaultdict(set)

#     for atom in model:
#         args = atom.arguments

#         if atom.name == 'body_literal':
#             rule_index = args[0].number
#             predicate = args[1].name
#             atom_args = args[3].arguments
#             atom_args = tuple(arg_lookup[arg] for arg in atom_args)
#             arity = len(atom_args)
#             body_literal = (predicate, atom_args, arity)
#             rule_index_to_body[rule_index].add(body_literal)

#         elif atom.name == 'head_literal':
#             rule_index = args[0].number
#             predicate = args[1].name
#             atom_args = args[3].arguments
#             atom_args = tuple(arg_lookup[arg] for arg in atom_args)
#             arity = len(atom_args)
#             head_literal = (predicate, atom_args, arity)
#             rule_index_to_head[rule_index] = head_literal

#         elif atom.name == 'direction_':
#             pred_name = args[0].name
#             arg_index = args[1].number
#             arg_dir_str = args[2].name

#             if arg_dir_str == 'in':
#                 arg_dir = '+'
#             elif arg_dir_str == 'out':
#                 arg_dir = '-'
#             else:
#                 raise Exception(f'Unrecognised argument direction "{arg_dir_str}"')
#             directions[pred_name][arg_index] = arg_dir

#         elif atom.name == 'before':
#             rule1 = args[0].number
#             rule2 = args[1].number
#             rule_index_ordering[rule1].add(rule2)

#     prog = []
#     rule_lookup = {}

#     # rules = set(rule_index_to_head.keys()).union(set(rule_index_to_body.keys()))
#     # for rule_index in rules:
#     #     head = None
#     #     if rule_index in rule_index_to_head:
#     for rule_index in rule_index_to_head:
#         head_pred, head_args, head_arity = rule_index_to_head[rule_index]
#         head_modes = tuple(directions[head_pred][i] for i in range(head_arity))
#         head = Literal(head_pred, head_args, head_modes)
#         body = set()
#         for (body_pred, body_args, body_arity) in rule_index_to_body[rule_index]:
#             body_modes = tuple(directions[body_pred][i] for i in range(body_arity))
#             body.add(Literal(body_pred, body_args, body_modes))
#         body = frozenset(body)
#         rule = head, body
#         prog.append((rule))
#         rule_lookup[rule_index] = rule

#     rule_ordering = defaultdict(set)
#     for r1_index, lower_rule_indices in rule_index_ordering.items():
#         r1 = rule_lookup[r1_index]
#         rule_ordering[r1] = set(rule_lookup[r2_index] for r2_index in lower_rule_indices)

#     return frozenset(prog), rule_ordering, directions


# class Generator:

#     def con_to_strings(self, con):
#          for grule in self.get_ground_rules((None, con)):
#             # print('grule', grule)
#             h, b = grule
#             rule = []
#             for sign, pred, args in b:
#                 if not sign:
#                     rule.append(f'not {pred}{args}')
#                 else:
#                     rule.append(f'{pred}{args}')
#             rule = ':- ' + ', '.join(sorted(rule)) + '.'
#             rule = rule.replace("'","")
#             rule = rule.replace('not clause(1,)','not clause(1)')
#             yield rule

#     def __init__(self, settings, grounder):
#         self.settings = settings
#         self.grounder = grounder

#         encoding = []
#         alan = pkg_resources.resource_string(__name__, "lp/alan.pl").decode()
#         encoding.append(alan)
#         with open(settings.bias_file) as f:
#             encoding.append(f.read())
#         encoding.append(f'max_clauses({settings.max_rules}).')
#         encoding.append(f'max_body({settings.max_body}).')
#         encoding.append(f'max_vars({settings.max_vars}).')

#         if self.settings.bkcons:
#             encoding.append(self.settings.bkcons)

#         encoding = '\n'.join(encoding)

#         solver = clingo.Control(["--heuristic=Domain"])
#         # solver = clingo.Control([])
#         # solver = clingo.Control(["-t2"])
#         solver.configuration.solve.models = 0
#         solver.add('base', [], encoding)
#         solver.ground([('base', [])])
#         self.solver = solver


#     def get_ground_rules(self, rule):
#         head, body = rule

#         # find bindings for variables in the rule
#         assignments = self.grounder.find_bindings(rule, self.settings.max_rules, self.settings.max_vars)

#         # keep only standard literals
#         body = tuple(literal for literal in body if not literal.meta)

#         # ground the rule for each variable assignment
#         return set(self.grounder.ground_rule((head, body), assignment) for assignment in assignments)


#     def build_generalisation_constraint(self, prog, rule_ordering={}):
#         prog = list(prog)
#         rule_index = {}
#         literals = []
#         for clause_number, rule in enumerate(prog):
#             rule_index[rule] = vo_clause(clause_number)
#             head, body = rule
#             clause_number = vo_clause(clause_number)
#             literals.append(Literal('head_literal', (clause_number, head.predicate, head.arity, tuple(vo_variable2(rule_number, v) for v in head.arguments))))

#             for body_literal in body:
#                 literals.append(Literal('body_literal', (clause_number, body_literal.predicate, body_literal.arity, tuple(vo_variable(v) for v in body_literal.arguments))))

#             for idx, var in enumerate(head.arguments):
#                 literals.append(eq(vo_variable(var), idx))

#             literals.append(body_size_literal(clause_number, len(body)))

#         for r1, higher_rules in rule_ordering.items():
#             r1v = rule_index[r1]
#             for r2 in higher_rules:
#                 r2v = rule_index[r2]
#                 literals.append(lt(r1v, r2v))

#         return tuple(literals)

#     # def build_specialisation_constraint(self, prog, rule_ordering={}):
#     #     prog = list(prog)
#     #     rule_index = {}
#     #     literals = []
#     #     for clause_number, rule in enumerate(prog):
#     #         rule_index[rule] = vo_clause(clause_number)
#     #         head, body = rule
#     #         clause_number = vo_clause(clause_number)
#     #         literals.append(Literal('head_literal', (clause_number, head.predicate, head.arity, tuple(vo_variable(v) for v in head.arguments))))

#     #         for body_literal in body:
#     #             literals.append(Literal('body_literal', (clause_number, body_literal.predicate, body_literal.arity, tuple(vo_variable(v) for v in body_literal.arguments))))

#     #         for idx, var in enumerate(head.arguments):
#     #             literals.append(eq(vo_variable(var), idx))
#     #         literals.append(lt(clause_number, len(prog)))
#     #     literals.append(Literal('clause', (len(prog), ), positive = False))

#     #     for r1, higher_rules in rule_ordering.items():
#     #         r1v = rule_index[r1]
#     #         for r2 in higher_rules:
#     #             r2v = rule_index[r2]
#     #             literals.append(lt(r1v, r2v))
#     #     return tuple(literals)

#     # # DOES NOT WORK WITH PI!!!
#     # def redundancy_constraint1(self, prog, rule_ordering={}):
#     #     prog = list(prog)
#     #     num_rec = 0
#     #     for rule in prog:
#     #         head, _body = rule
#     #         if rule_is_recursive(rule):
#     #             num_rec += 1

#     #     rule_index = {}
#     #     literals = []

#     #     for rule_number, rule in enumerate(prog):
#     #         k = rule_number
#     #         rule_index[rule] = vo_clause(rule_number)
#     #         head, body = rule
#     #         rule_number = vo_clause(rule_number)
#     #         literals.append(Literal('head_literal', (rule_number, head.predicate, head.arity, tuple(vo_variable2(k, v) for v in head.arguments))))

#     #         for body_literal in body:
#     #             literals.append(Literal('body_literal', (rule_number, body_literal.predicate, body_literal.arity, tuple(vo_variable2(k, v) for v in body_literal.arguments))))

#     #         for idx, var in enumerate(head.arguments):
#     #             literals.append(eq(vo_variable2(k, var), idx))
#     #         literals.append(gteq(rule_number, 1))

#     #         literals.append(Literal('recursive_clause',(rule_number, head.predicate, head.arity)))
#     #         literals.append(Literal('num_recursive', (head.predicate, 1)))

#     #     for r1, higher_rules in rule_ordering.items():
#     #         r1v = rule_index[r1]
#     #         for r2 in higher_rules:
#     #             r2v = rule_index[r2]
#     #             literals.append(lt(r1v, r2v))
#     #     return tuple(literals)


#     # def redundancy_constraint2(self, prog, rule_ordering={}):
#     #     prog = list(prog)
#     #     rule_index = {}
#     #     # literals = []
#     #     lits_num_rules = defaultdict(int)
#     #     lits_num_recursive_rules = defaultdict(int)
#     #     for rule in prog:
#     #         head, _ = rule
#     #         lits_num_rules[head.predicate] += 1
#     #         if rule_is_recursive(rule):
#     #             lits_num_recursive_rules[head.predicate] += 1

#     #     recursively_called = set()
#     #     while True:
#     #         something_added = False
#     #         for rule in prog:
#     #             head, body = rule
#     #             is_rec = rule_is_recursive(rule)
#     #             for body_literal in body:
#     #                 if body_literal.predicate not in lits_num_rules:
#     #                     continue
#     #                 if (body_literal.predicate != head.predicate and is_rec) or (head.predicate in recursively_called):
#     #                     something_added |= not body_literal.predicate in recursively_called
#     #                     recursively_called.add(body_literal.predicate)
#     #         if not something_added:
#     #             break

#     #     for lit in lits_num_rules.keys() - recursively_called:
#     #         literals = []

#     #         for clause_number, rule in enumerate(prog):
#     #             k = clause_number
#     #             rule_index[rule] = vo_clause(clause_number)
#     #             head, body = rule
#     #             clause_number = vo_clause(clause_number)
#     #             literals.append(Literal('head_literal', (clause_number, head.predicate, head.arity, tuple(vo_variable2(k, v) for v in head.arguments))))

#     #             for body_literal in body:
#     #                 literals.append(Literal('body_literal', (clause_number, body_literal.predicate, body_literal.arity, tuple(vo_variable2(k,v) for v in body_literal.arguments))))

#     #             for idx, var in enumerate(head.arguments):
#     #                 literals.append(eq(vo_variable2(k,var), idx))

#     #         for other_lit, num_clauses in lits_num_rules.items():
#     #             if other_lit == lit:
#     #                 continue
#     #             literals.append(Literal('num_clauses', (other_lit, num_clauses)))
#     #         num_recursive = lits_num_recursive_rules[lit]

#     #         literals.append(Literal('num_recursive', (lit, num_recursive)))

#     #         return tuple(literals)


# def vo_variable(variable):
#     return ConstVar(f'{variable}', 'Variable')

# def vo_variable2(rule, variable):
#     # print(type(rule))
#     # return ConstVar(f'{rule}_{variable}', 'Variable')
#     key = f'V_{rule}_{variable}'
#     # return ConstVar(key, 'Variable')
#     # print(key, rule, variable)
#     # return ConstVar(f'{variable}', 'Variable')
#     return ConstVar(key, 'Variable')

# def alldiff(args):
#     return Literal('AllDifferent', args, meta=True)

# # def lt(rule, var, value):
# #     return Literal('<', (rule, var, value), meta=True)
# def lt(a, b):
#     return Literal('<', (a, b), meta=True)

# def eq(rule, var, value):
#     return Literal('==', (rule, var, value), meta=True)

# # def gteq(rule, var, value):
#     # return Literal('>=', (rule, var, value), meta=True)

# def vo_clause(variable):
#     return ConstVar(f'Rule{variable}', 'Clause')

# def body_size_literal(clause_var, body_size):
#     return Literal('body_size', (clause_var, body_size))

# def alldiff(args):
#     return Literal('AllDifferent', args, meta=True)

# def find_all_vars(body):
#     all_vars = set()
#     for literal in body:
#         for arg in literal.arguments:
#             if isinstance(arg, ConstVar):
#                 all_vars.add(arg)
#             elif isinstance(arg, tuple):
#                 for t_arg in arg:
#                     if isinstance(t_arg, ConstVar):
#                         all_vars.add(t_arg)
#     return all_vars

# # AC: When grounding constraint rules, we only care about the vars and the constraints, not the actual literals
# def grounding_hash(body, all_vars):
#     cons = set()
#     for lit in body:
#         if lit.meta:
#             cons.add((lit.predicate, lit.arguments))
#     return hash((frozenset(all_vars), frozenset(cons)))

# class Grounder():
#     def __init__(self):
#         self.seen_assignments = {}

#     # function to ground first-order constraints
#     def find_bindings(self, rule, max_rules, max_vars):
#         _, body = rule
#         # TODO: add back
#         # all_vars = find_all_vars(body)
#         # if len(all_vars) == 0:
#         #     return [{}]

#         # k = grounding_hash(body, all_vars)
#         # if k in self.seen_assignments:
#         #     return self.seen_assignments[k]

#         print('HELLO!!!')

#         # map each rule and var_var in the program to an integer
#         rule_id_to_var = {v:i for i,v in enumerate(var for var in all_vars if var.type == 'Clause')}

#         encoding = []
#         encoding.add(f'#const max_rules={max_rules}.')
#         encoding.add(f'#const max_vars={max_vars}.')

#         # find all variables for each rule
#         rule_vars = defaultdict(set)
#         for var in all_vars:
#             if var.type == 'Variable':
#                 _k, rule_id, _ = var.name.split('_')
#                 rule_vars[rule_id].add(var)

#         rule_var_to_int = {}
#         rule_var_lookup = {}
#         for rule_id, xs in rule_vars.items():
#             rule_var = rule_to_var[rule_id]
#             for i, var in enumerate(xs):
#                 encoding.add(f'rule_var({rule_var},{i}).')
#                 rule_var_lookup[(rule_var, i)] = var
#                 rule_var_to_int[var] = i

#         encoding.append(
#         """\
#             #show bind_rule/2.
#             #show bind_var/3.

#             % bind a rule_id to a value
#             {bind_rule(Rule,Value)}:-
#                 rule_var(Rule,_),
#                 Value=0..max_rules-1.
#             {bind_var(Rule,Var,Value)}:-
#                 rule_var(Rule,Var),
#                 Value=0..max_vars-1.

#             % every rule must be bound to exactly one value
#             :-
#                 rule_var(Rule,_),
#                 #count{Value: bind_rule(Rule,Value} != 1.
#             % for each rule, each var must be bound to exactly one value
#             :-
#                 rule_var(Rule,Var),
#                 #count{Value: bind_var(Rule,Var,Value} != 1.
#             % a rule value cannot be bound to more than one rule
#             :-
#                 Value=0..num_rules-1.
#                 #count{Rule : bind_rule(Rule,Value)} > 1.
#             % a var value cannot be bound to more than one var per rule
#             :-
#                 rule_var(Rule,_),
#                 Value=0..max_vars-1,
#                 #count{Var : bind_var(Rule,Var,Value)} > 1.
#         """)

#         # rule_var_lookup[(rule_var, i)] = var
#         # rule_var_to_int[var] = i
#         # add constraints to the ASP program based on the AST thing
#         for lit in body:
#             if not lit.meta:
#                 continue
#             if lit.predicate == '==':
#                 ruleid, varid, val = lit.arguments
#                 var_var = ruleid_and_varid_to_var[(ruleid, varid)]
#                 encoding.append(f':- not bind_var({rule_var},{var_var},{val}).')
#             # elif lit.predicate == '>=':
#             #     var, val = lit.arguments
#             #     var = c_vars[var]
#             #     for i in range(val):
#             #         encoding.append(f':- c_var({var},{i}).')
#             elif lit.predicate == '<':
#                 # rule, var, val = lit.arguments
#                 a, b = lit.arguments
#                 # b = lit.arguments[1]
#                 if type(val) == int:
#                 # ABSOLUTE HACK
#                     var1 = c_vars[a]
#                     encoding.append(f':- bind_rule({var1},Val1), Val1 >= {b}.')
#                     # pass
#                 else:
#                     var1 = c_vars[a]
#                     var2 = c_vars[b]
#                     encoding.append(f':- bind_rule({var1},Val1), bind_rule({var2},Val2), Val1>=Val2.')

#         encoding = '\n'.join(encoding)
#         solver = clingo.Control()
#         # ask for all models
#         solver.configuration.solve.models = 0
#         solver.add('base', [], encoding)
#         solver.ground([("base", [])])

#         out = []

#         def on_model(m):
#             xs = m.symbols(shown = True)
#             # map a variable to a program variable
#             assignment = {}
#             for x in xs:
#                 name = x.name
#                 args = x.arguments
#                 if name == 'bind_var':
#                     rule = args[0].number
#                     var = args[1].number
#                     val = args[2].number
#                     assignment[var_var_lookup[var]] = val
#                 else:
#                     rule = args[0].number
#                     val = args[1].number
#                     assignment[rule_var_lookup[rule]] = val
#             out.append(assignment)

#         solver.solve(on_model=on_model)
#         self.seen_assignments[k] = out
#         return out

#     def ground_literal(self, literal, assignment):
#         ground_args = []
#         for arg in literal.arguments:
#             if arg in assignment:
#                 ground_args.append(assignment[arg])
#             # handles tuples of ConstVars
#             # TODO: AC: EXPLAIN BETTER
#             elif isinstance(arg, tuple):
#                 ground_t_args = []
#                 # AC: really messy
#                 for t_arg in arg:
#                     if t_arg in assignment:
#                         ground_t_args.append(assignment[t_arg])
#                     else:
#                         ground_t_args.append(t_arg)
#                 ground_args.append(tuple(ground_t_args))
#             else:
#                 ground_args.append(arg)
#         return literal.positive, literal.predicate, tuple(ground_args)

#     def ground_rule(self, rule, assignment):
#         head, body = rule
#         ground_head = None
#         if head:
#             ground_head = self.ground_literal(head, assignment)
#         ground_body = frozenset(self.ground_literal(literal, assignment) for literal in body)
#         return ground_head, ground_body

