"""Shared test helpers: a compact fragment builder and a scalar reference interpreter.

The reference interpreter is written on plain Python integers, one vector at a time, on
purpose. It shares no code with the numpy evaluator, so agreement between the two is
evidence that both follow the same semantics.
"""

from superopt.ir import Arg, Const, Fragment, Input, Inst, mask, to_signed, validate


def const(value: int, width: int) -> Const:
    """Build a constant from a possibly negative Python integer."""
    return Const(value & mask(width), width)


def build(input_widths: list[int], rows: list[tuple], external_uses: tuple[str, ...] = ()):
    """Build a validated fragment from rows of (op, width, operand, ...).

    Inputs are named a0, a1, ... and instructions t0, t1, ... in row order. An icmp is
    written as "icmp.<pred>". The last row is the root.
    """
    inputs = []
    for index, width in enumerate(input_widths):
        inputs.append(Input(f"a{index}", width))
    insts = []
    for index, row in enumerate(rows):
        op = row[0]
        pred = None
        if op.startswith("icmp."):
            pred = op.split(".", 1)[1]
            op = "icmp"
        insts.append(Inst(f"t{index}", op, row[1], tuple(row[2:]), pred))
    fragment = Fragment(
        inputs=tuple(inputs),
        insts=tuple(insts),
        root=insts[-1].name,
        external_uses=external_uses,
    )
    validate(fragment)
    return fragment


def leaf(input_widths: list[int], root: Arg) -> Fragment:
    """Build a fragment with no instructions whose root is an input or a constant."""
    inputs = []
    for index, width in enumerate(input_widths):
        inputs.append(Input(f"a{index}", width))
    fragment = Fragment(inputs=tuple(inputs), insts=(), root=root)
    validate(fragment)
    return fragment


def reference_binary(op: str, width: int, left: int, right: int) -> tuple[int, bool]:
    """Return (value, poison) for a binary op on unsigned bit patterns."""
    if op == "add":
        return (left + right) & mask(width), False
    if op == "sub":
        return (left - right) & mask(width), False
    if op == "mul":
        return (left * right) & mask(width), False
    if op == "and":
        return left & right, False
    if op == "or":
        return left | right, False
    if op == "xor":
        return left ^ right, False
    if right >= width:
        return 0, True
    if op == "shl":
        return (left << right) & mask(width), False
    if op == "lshr":
        return left >> right, False
    if op == "ashr":
        return (to_signed(left, width) >> right) & mask(width), False
    raise ValueError(op)


def reference_icmp(pred: str, width: int, left: int, right: int) -> int:
    if pred[0] == "s":
        left = to_signed(left, width)
        right = to_signed(right, width)
    kind = pred[-2:]
    if pred == "eq":
        return int(left == right)
    if pred == "ne":
        return int(left != right)
    if kind == "gt":
        return int(left > right)
    if kind == "ge":
        return int(left >= right)
    if kind == "lt":
        return int(left < right)
    if kind == "le":
        return int(left <= right)
    raise ValueError(pred)


def reference_eval(
    fragment: Fragment, input_values: list[int], input_poison: list[bool]
) -> tuple[int, bool]:
    """Evaluate a fragment on one vector. Returns (value, poison), value 0 under poison."""
    values: dict[str, tuple[int, bool]] = {}
    widths: dict[str, int] = {}
    for index, fragment_input in enumerate(fragment.inputs):
        if input_poison[index]:
            values[fragment_input.name] = (0, True)
        else:
            values[fragment_input.name] = (input_values[index], False)
        widths[fragment_input.name] = fragment_input.width

    def lookup(arg: Arg) -> tuple[int, bool, int]:
        if isinstance(arg, Const):
            return arg.value, False, arg.width
        value, poison = values[arg]
        return value, poison, widths[arg]

    for inst in fragment.insts:
        operands = []
        for arg in inst.args:
            operands.append(lookup(arg))
        if inst.op == "select":
            condition_value, condition_poison, _ = operands[0]
            if condition_poison:
                result = (0, True)
            elif condition_value == 1:
                result = (operands[1][0], operands[1][1])
            else:
                result = (operands[2][0], operands[2][1])
        elif inst.op in ("zext", "sext", "trunc"):
            value, poison, source_width = operands[0]
            if poison:
                result = (0, True)
            elif inst.op == "zext":
                result = (value, False)
            elif inst.op == "sext":
                result = (to_signed(value, source_width) & mask(inst.width), False)
            else:
                result = (value & mask(inst.width), False)
        else:
            left_value, left_poison, operand_width = operands[0]
            right_value, right_poison, _ = operands[1]
            if left_poison or right_poison:
                result = (0, True)
            elif inst.op == "icmp":
                result = (reference_icmp(inst.pred, operand_width, left_value, right_value), False)
            else:
                value, poison = reference_binary(inst.op, inst.width, left_value, right_value)
                if poison:
                    value = 0
                result = (value, poison)
        if result[1]:
            result = (0, True)
        values[inst.name] = result
        widths[inst.name] = inst.width

    value, poison, _ = lookup(fragment.root)
    return value, poison
