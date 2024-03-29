from ast import *
from io import StringIO
import sys
import re


def check_file(file, parts, max_score=2, deductions=None):
    if deductions is None:
        deductions = {"print": False}
    with open(file) as fp:
        return check_answer(fp.read(), parts, max_score, deductions)


def check_answer(answer, parts, max_score, deductions):
    answer = answer + f"\n\n({part_letter(len(parts))})"
    return {part_letter(i): check_part(answer, i, *part, max_score, deductions)
            for i, part in enumerate(parts)}


def check_part(answer, i, expected, ops, max_score, deductions):
    pattern = (f"\\({part_letter(i)}\\).*?$\n"
               " *(.*?)$\n+"
               f"(?:\\({part_letter(i + 1)}\\))")

    match = re.search(pattern, answer, flags=re.MULTILINE | re.DOTALL)
    if not match:
        return 0, ["no answer found"]

    return check_expr(match.group(1), expected, ops, max_score, deductions)


def part_letter(i): return chr(ord('a') + i)


def check_expr(expr, expected, ops, max_score, deductions):
    try:
        parsed = parse(expr)

        if len(parsed.body) != 1:
            return 0, ["multiple statements"]

        parsed = parsed.body[0]
        if not isinstance(parsed, Expr):
            return 0, ["invalid statement"]

        deduct_print = isinstance(parsed.value, Call) \
            and isinstance(parsed.value.func, Name) \
            and parsed.value.func.id == "print"
        if deduct_print:
            call = parsed.value
            if len(call.args) == 1 and len(call.keywords) == 0:
                val = eval_out(get_source_segment(expr, call.args[0]))[0]
            else:
                val = eval_out(expr)[1].strip()
        else:
            val = eval_out(expr)[0]

        expected_ty = type(expected)
        if str(val) == str(expected) and not isinstance(val, expected_ty):
            score = max_score - 1
            reasons = [
                f"incorrect expression type {type(val).__name__}, "
                f"expecting {expected_ty.__name__}"]
        elif val != expected:
            return 0, [f"incorrect expression value {repr(val)}, "
                       f"expecting {repr(expected)}"]
        else:
            score, reasons = max_score, []

        for name, op in ops.items():
            deduction, reason = -1, [f"'{name}' unused"]
            for node in walk(parsed):
                try:
                    deduction_reason = op(expr, node)
                    if deduction_reason is None:
                        continue
                    if deduction_reason[0] >= deduction:
                        deduction, reason = deduction_reason
                except Exception as e:
                    print(f"internal {name}: {repr(e)}", file=sys.stderr)

            score += deduction
            if reason is not None:
                reasons += reason
            if score <= 0:
                break

        if deduct_print and not deductions.get("print", False) and score > 0:
            if "print" in deductions:
                deductions["print"] = True
            score -= 1
            reasons.append("print statement")

        return max(0, score), reasons
    except Exception as e:
        return 0, [f"invalid expression ({e})"]


def eval_out(str):
    old_stdout = sys.stdout
    sys.stdout = new_stdout = StringIO()
    try:
        out = eval(str, {})
    finally:
        sys.stdout = old_stdout
    return out, new_stdout.getvalue()


def node_type(src, node):
    seg = get_source_segment(src, node)
    if seg is None:
        raise Exception(node)
    return type(eval_out(seg)[0])


def any_of(*ops):
    def f(src, node):
        deductions = [op(src, node) for op in ops]
        deductions = [x for x in deductions if x is not None]
        if deductions:
            return sorted(deductions)[-1]
    return f


def arg_types(src, args, name, tys):
    if tys is None:
        return

    if len(args) != len(tys):
        return -1, [f"expecting call to {name} with {len(tys)} arguments"]

    reasons, deductions = [], None
    for i, (arg, ty) in enumerate(zip(args, tys)):
        if ty is None:
            continue
        arg_ty = node_type(src, arg)
        if arg_ty is not ty:
            deductions = -1
            reasons.append(f"argument {i} of {name} has type "
                           f"{arg_ty.__name__}, expecting {ty.__name__}")

    if deductions is not None:
        return reasons, deductions


def function_call(name, tys):
    def f(src, node):
        if not isinstance(node, Call):
            return
        if not isinstance(node.func, Name):
            return

        if node.func.id != name:
            return -1, [f"call to {node.func.id}, expecting {name}"]

        ret = arg_types(src, node.args, name, tys)
        if ret is not None:
            return ret

        return 0, []
    return f


def method_call(obj_ty, name, tys=None):
    def f(src, node):
        if not isinstance(node, Call):
            return
        if not isinstance(node.func, Attribute):
            return

        if node.func.attr != name:
            return -1, [f"method call to {node.func.attr}, expecting {name}"]

        func_ty = node_type(src, node.func.value)
        if obj_ty is not None and func_ty is not obj_ty:
            return -1, [f"object in method call to {name} of type "
                        f"{func_ty.__name__}, expecting {obj_ty}"]

        ret = arg_types(src, node.args, name, tys)
        if ret is not None:
            return ret

        return 0, []
    return f


def constant(val=None, ty=None):
    def f(src, node):
        if not isinstance(node, Constant):
            return

        if val is not None:
            if node.value != val or not isinstance(node.value, type(val)):
                return -1, [f"incorrect constant value {node.value}, "
                            f"expecting {val}"]

        if ty is not None:
            if not isinstance(node.value, ty):
                return -1, [f"incorrect constant value {val_ty} of type "
                            f"{type(node.value).__name__}, expecting {ty}"]

        return 0, []
    return f


def slice(ty=None):
    def f(src, node):
        if not isinstance(node, Subscript):
            return
        if not isinstance(node.slice, Slice):
            return -1, ["used index, expecting slice"]

        if ty is not None:
            slc_ty = node_type(src, node.value)
            if slc_ty is not ty:
                return -1, [f"slice of type {slc_ty.__name__}, "
                            f"expecting {ty.__name__}"]

        return 0, []
    return f


def index(ty):
    def f(src, node):
        if not isinstance(node, Subscript):
            return
        if isinstance(node.slice, Slice):
            return -1, ["used slice, expecting index"]

        if ty is not None:
            val_ty = node_type(src, node.value)
            if val_ty is not ty:
                return -1, [f"index of type {val_ty.__name__}, "
                            f"expecting {ty.__name__}"]

        return 0, []
    return f


UN_OPS = {type(ctor()): op for op, ctor in [
    ("+", UAdd),
    ("-", USub),
    ("not", Not),
    ("~", Invert)
]}


def unop(op, op_ty=None, exp_ty=None):
    def f(src, node):
        if not isinstance(node, UnaryOp):
            return

        node_op = UN_OPS.get(type(node.op))
        if node_op != op:
            return

        if exp_ty is not None:
            node_ty = node_type(node)
            if node_ty is not exp_ty:
                return -1, [f"{op} use results in type {node_ty.__name__}, "
                            f"expecting {exp_ty.__name__}"]

        if op_ty is not None:
            val_ty = node_type(src, node.operand)
            if val_ty is not op_ty:
                return -1, [f"operand of {op} has type {val_ty.__name__}, "
                            f"expecting {op_ty.__name__}"]

        return 0, []
    return f


BIN_OPS = {type(ctor()): op for op, ctor in [
    ("+", Add),
    ("-", Sub),
    ("*", Mult),
    ("/", Div),
    ("//", FloorDiv),
    ("%", Mod),
    ("**", Pow),
    ("<<", LShift),
    (">>", RShift),
    ("|", BitOr),
    ("^", BitXor),
    ("&", BitAnd)
]}


def binop(op, left_ty=None, right_ty=None, exp_ty=None):
    def f(src, node):
        if not isinstance(node, BinOp):
            return

        node_op = BIN_OPS.get(type(node.op))
        if node_op != op:
            return

        if exp_ty is not None:
            node_ty = node_type(src, node)
            if node_ty is not exp_ty:
                return -1, [f"{op} use results in type {node_ty.__name__}, "
                            f"expecting {exp_ty.__name__}"]

        deductions, reasons = 0, []
        for ty, attr in [(left_ty, "left"), (right_ty, "right")]:
            if ty is None:
                continue
            val_ty = node_type(src, getattr(node, attr))
            if val_ty is not ty:
                deductions = -1
                reasons.append(
                    f"{attr} operand of {op} has type {val_ty.__name__}, "
                    f"expecting {ty.__name__}")

        return deductions, reasons
    return f


BOOL_OPS = {type(ctor()): op for op, ctor in [
    ("and", And),
    ("or", Or)
]}


def boolop(op, left_ty=None, right_ty=None, exp_ty=None):
    def f(src, node):
        if not isinstance(node, BoolOp):
            return

        node_op = BOOL_OPS.get(type(node.op))
        if node_op != op:
            return

        if exp_ty is not None:
            node_ty = node_type(node)
            if node_ty is not exp_ty:
                return -1, [f"{op} use results in type {node_ty.__name__}, "
                            f"expecting {exp_ty.__name__}"]

        deductions, reasons = 0, []
        for val, (ty, attr) in zip(node.values,
                                   [(left_ty, "left"), (right_ty, "right")]):
            if ty is None:
                continue
            val_ty = node_type(src, val)
            if val_ty is not ty:
                deductions = -1
                reasons.append(
                    f"{attr} operand of {op} has type {val_ty.__name__}, "
                    f"expecting {ty.__name__}")

        return deductions, reasons
    return f


CMP_OPS = {type(ctor()): op for op, ctor in [
    ("==", Eq),
    ("!=", NotEq),
    ("<", Lt),
    ("<=", LtE),
    (">", Gt),
    (">=", GtE),
    ("is", Is),
    ("is not", IsNot),
    ("in", In),
    ("not in", NotIn)
]}


def compare(op, left_ty=None, right_ty=None):
    def helper(src, left, right, cmp):
        cmp_op = CMP_OPS.get(type(cmp))
        if cmp_op != op:
            return -1, [f"used {cmp_op}, expecting {op}"]

        deductions, reasons = 0, []
        for ty, attr, n in [(left_ty, "left", left),
                            (right_ty, "right", right)]:
            if ty is None:
                continue
            val_ty = node_type(src, n)
            if val_ty is not ty:
                deductions = -1
                reasons.append(
                    f"{attr} operand of {op} has type {val_ty.__name__}, "
                    f"expecting {ty.__name__}")

        return deductions, reasons

    def f(src, node):
        if not isinstance(node, Compare):
            return

        deductions = [helper(src, *x)
                      for x in zip([node.left] + node.comparators,
                                   node.comparators,
                                   node.ops)]
        deductions = [x for x in deductions if x is not None]

        if deductions:
            return sorted(deductions)[-1]
    return f


def f_string(with_pattern=True):
    def f(src, node):
        if not isinstance(node, JoinedStr):
            return

        if with_pattern and not any(lambda x: isinstance(x, FormattedValue),
                                    node.values):
            return -1, ["f-string has no formatted values"]

        return 0, []
    return f


def instance(ctor):
    def f(src, node):
        if not isinstance(node, ctor):
            return
        return 0, []


STR_INDEX = index(str)
LIST_INDEX = index(list)
TUPLE_INDEX = index(tuple)
INDEX = any_of(STR_INDEX, LIST_INDEX, TUPLE_INDEX)
INDEX_METHOD = method_call(None, "index")

STR_SLICE = slice(str)
LIST_SLICE = slice(list)
TUPLE_SLICE = slice(tuple)
SLICE = any_of(STR_SLICE, LIST_SLICE, TUPLE_SLICE)

STR_CONCAT = binop("+", exp_ty=str)
LIST_CONCAT = binop("+", exp_ty=list)
TUPLE_CONCAT = binop("+", exp_ty=tuple)
CONCAT = any_of(STR_CONCAT, LIST_CONCAT, TUPLE_CONCAT)

STR_REPLICATE = binop("*", exp_ty=str)
LIST_REPLICATE = binop("*", exp_ty=list)
TUPLE_REPLICATE = binop("*", exp_ty=tuple)
REPLICATE = any_of(STR_REPLICATE, LIST_REPLICATE, TUPLE_REPLICATE)

DICT_LOOKUP = any_of(
    method_call(dict, "get", [None]),
    method_call(dict, "get", [None, None]),
    index(dict)
)

INT_ADD = binop("+", exp_ty=int)
FLOAT_ADD = binop("+", exp_ty=float)
ADD = any_of(INT_ADD, FLOAT_ADD)
INT_SUB = binop("-", exp_ty=int)
FLOAT_SUB = binop("-", exp_ty=float)
SUB = any_of(INT_SUB, FLOAT_SUB)
INT_MULT = binop("*", exp_ty=int)
FLOAT_MULT = binop("*", exp_ty=float)
MULT = any_of(INT_MULT, FLOAT_MULT)
INT_DIV = binop("//", exp_ty=int)
FLOOR_DIV = binop("//")
FLOAT_DIV = binop("/")
DIV = any_of(FLOOR_DIV, FLOAT_DIV)
INT_MOD = binop("%", exp_ty=int)
FLOAT_MOD = binop("%", exp_ty=float)
MOD = any_of(INT_MOD, FLOAT_MOD)
POW = binop("**")

LEN = function_call("len", [None])
SORTED = function_call("sorted", [None])

LIST = instance(List)
TUPLE = instance(Tuple)
SET = instance(Set)
DICT = instance(Dict)

STRING = constant(ty=str)
F_STRING = f_string()
F_STRING_NO_PATTERN = f_string(with_pattern=False)
STR_FORMAT = binop("%", exp_ty=str)

LIST_COMPREHENSION = instance(ListComp)
SET_COMPREHENSION = instance(SetComp)
DICT_COMPREHENSION = instance(DictComp)
GENERATOR = instance(GeneratorExp)

LAMBDA = instance(Lambda)
TERNARY = instance(IfExp)
