"""Fragment data model: dataclasses, validation, a text printer, and the JSON contract.

A fragment is a small straight-line integer computation: typed inputs, instructions in
topological order, and one root value. Harvested sources and synthesized targets use the
same classes. A target may have no instructions at all, in which case its root is an input
name or a constant.
"""

from dataclasses import dataclass

from superopt import config

CONTRACT_VERSION = 1


class FragmentError(ValueError):
    """A fragment is malformed: bad JSON shape, bad types, or bad structure."""


@dataclass(frozen=True)
class Const:
    """An integer constant. `value` is the unsigned bit pattern, 0 <= value < 2**width."""

    value: int
    width: int


@dataclass(frozen=True)
class Input:
    name: str
    width: int


# An operand is either the name of an input or instruction, or a constant.
Arg = str | Const


@dataclass(frozen=True)
class Inst:
    """One instruction. `width` is the result width. `pred` is set only for icmp."""

    name: str
    op: str
    width: int
    args: tuple[Arg, ...]
    pred: str | None = None


@dataclass(frozen=True)
class Location:
    """Where a harvested fragment came from."""

    corpus: str
    file: str
    function: str
    block: int


@dataclass(frozen=True)
class Fragment:
    inputs: tuple[Input, ...]
    insts: tuple[Inst, ...]
    root: Arg
    external_uses: tuple[str, ...] = ()
    stripped_flags: tuple[str, ...] = ()
    location: Location | None = None
    orig_id: int | None = None


def mask(width: int) -> int:
    """All ones at the given width."""
    return (1 << width) - 1


def to_signed(value: int, width: int) -> int:
    """Read an unsigned bit pattern as a two's complement number."""
    if value >= 1 << (width - 1):
        return value - (1 << width)
    return value


def value_widths(fragment: Fragment) -> dict[str, int]:
    """Map every input and instruction name to its width."""
    widths = {}
    for fragment_input in fragment.inputs:
        widths[fragment_input.name] = fragment_input.width
    for inst in fragment.insts:
        widths[inst.name] = inst.width
    return widths


def arg_width(arg: Arg, widths: dict[str, int]) -> int:
    if isinstance(arg, Const):
        return arg.width
    return widths[arg]


def root_width(fragment: Fragment) -> int:
    return arg_width(fragment.root, value_widths(fragment))


def inst_by_name(fragment: Fragment) -> dict[str, Inst]:
    table = {}
    for inst in fragment.insts:
        table[inst.name] = inst
    return table


def check_const(const: Const, context: str) -> None:
    if const.width not in config.ALLOWED_WIDTHS:
        raise FragmentError(f"{context}: constant width {const.width} is not allowed")
    if const.value < 0 or const.value > mask(const.width):
        raise FragmentError(f"{context}: constant {const.value} does not fit i{const.width}")


def check_arity(inst: Inst, expected: int) -> None:
    if len(inst.args) != expected:
        raise FragmentError(
            f"{inst.name}: {inst.op} takes {expected} operands, got {len(inst.args)}"
        )


def check_inst_types(inst: Inst, widths: dict[str, int]) -> None:
    """Check operand count and operand widths for one instruction."""
    operand_widths = []
    for arg in inst.args:
        operand_widths.append(arg_width(arg, widths))

    if inst.op in config.BINARY_OPS:
        check_arity(inst, 2)
        for operand_width in operand_widths:
            if operand_width != inst.width:
                raise FragmentError(
                    f"{inst.name}: {inst.op} operand is i{operand_width}, result is i{inst.width}"
                )
        return

    if inst.op == "icmp":
        check_arity(inst, 2)
        if inst.width != 1:
            raise FragmentError(f"{inst.name}: icmp result must be i1")
        if inst.pred not in config.ICMP_PREDICATES:
            raise FragmentError(f"{inst.name}: unknown icmp predicate {inst.pred!r}")
        if operand_widths[0] != operand_widths[1]:
            raise FragmentError(f"{inst.name}: icmp operands have different widths")
        return

    if inst.op == "select":
        check_arity(inst, 3)
        if operand_widths[0] != 1:
            raise FragmentError(f"{inst.name}: select condition must be i1")
        if operand_widths[1] != inst.width or operand_widths[2] != inst.width:
            raise FragmentError(f"{inst.name}: select arms must match the result width")
        return

    if inst.op in ("zext", "sext"):
        check_arity(inst, 1)
        if operand_widths[0] >= inst.width:
            raise FragmentError(f"{inst.name}: {inst.op} must widen its operand")
        return

    if inst.op == "trunc":
        check_arity(inst, 1)
        if operand_widths[0] <= inst.width:
            raise FragmentError(f"{inst.name}: trunc must narrow its operand")
        return

    raise FragmentError(f"{inst.name}: unknown op {inst.op!r}")


def reachable_from_root(fragment: Fragment) -> set[str]:
    """Names of all inputs and instructions the root depends on, including the root."""
    table = inst_by_name(fragment)
    reached: set[str] = set()
    pending: list[str] = []
    if isinstance(fragment.root, str):
        pending.append(fragment.root)
    while pending:
        name = pending.pop()
        if name in reached:
            continue
        reached.add(name)
        inst = table.get(name)
        if inst is None:
            continue
        for arg in inst.args:
            if isinstance(arg, str):
                pending.append(arg)
    return reached


def validate(fragment: Fragment, harvested: bool = False) -> None:
    """Raise FragmentError unless the fragment is well formed.

    Checks: allowed widths, unique names, operands defined before use, operand types,
    root placement, and no dead instructions. With `harvested` set, also require what
    the harvester guarantees: at least one instruction and every input used.
    """
    widths: dict[str, int] = {}
    for fragment_input in fragment.inputs:
        if fragment_input.width not in config.ALLOWED_WIDTHS:
            raise FragmentError(f"input {fragment_input.name}: width is not allowed")
        if fragment_input.name in widths:
            raise FragmentError(f"duplicate name {fragment_input.name}")
        widths[fragment_input.name] = fragment_input.width

    for inst in fragment.insts:
        if inst.width not in config.ALLOWED_WIDTHS:
            raise FragmentError(f"{inst.name}: width {inst.width} is not allowed")
        if inst.name in widths:
            raise FragmentError(f"duplicate name {inst.name}")
        if inst.op != "icmp" and inst.pred is not None:
            raise FragmentError(f"{inst.name}: only icmp carries a predicate")
        for arg in inst.args:
            if isinstance(arg, Const):
                check_const(arg, inst.name)
            elif arg not in widths:
                raise FragmentError(f"{inst.name}: operand {arg} is used before it is defined")
        check_inst_types(inst, widths)
        widths[inst.name] = inst.width

    if isinstance(fragment.root, Const):
        check_const(fragment.root, "root")
        if fragment.insts:
            raise FragmentError("a constant root cannot have instructions")
    elif fragment.insts:
        if fragment.root != fragment.insts[-1].name:
            raise FragmentError("the root must be the last instruction")
    elif fragment.root not in widths:
        raise FragmentError(f"root {fragment.root} is not defined")

    reached = reachable_from_root(fragment)
    inst_names = set()
    for inst in fragment.insts:
        inst_names.add(inst.name)
        if inst.name not in reached:
            raise FragmentError(f"{inst.name}: instruction is not used by the root")

    for name in fragment.external_uses:
        if name not in inst_names:
            raise FragmentError(f"external use {name} is not an instruction")
        if name == fragment.root:
            raise FragmentError("the root cannot be listed as an external use")

    for flag in fragment.stripped_flags:
        if flag not in config.KNOWN_FLAGS:
            raise FragmentError(f"unknown stripped flag {flag!r}")

    if harvested:
        if not fragment.insts:
            raise FragmentError("a harvested fragment needs at least one instruction")
        for fragment_input in fragment.inputs:
            if fragment_input.name not in reached:
                raise FragmentError(f"input {fragment_input.name} is never used")


def format_const(const: Const) -> str:
    """Print a constant with its width, for example `-1:i32`, `255:i8`, `0xff00ff:i32`.

    Small negative numbers print signed and large values print in hex, because that is
    how masks and decrements are easiest to read. The width is always printed, so the
    text identifies the constant exactly.
    """
    signed_value = to_signed(const.value, const.width)
    if const.width > 1 and -65536 <= signed_value < 0:
        return f"{signed_value}:i{const.width}"
    if const.value >= 65536:
        return f"{const.value:#x}:i{const.width}"
    return f"{const.value}:i{const.width}"


def arg_text(arg: Arg, table: dict[str, Inst], marked: frozenset[str]) -> str:
    """Print one operand as a nested expression."""
    if isinstance(arg, Const):
        return format_const(arg)
    inst = table.get(arg)
    if inst is None:
        return arg
    parts = []
    for operand in inst.args:
        parts.append(arg_text(operand, table, marked))
    head = inst.op
    if inst.op == "icmp":
        head = f"icmp.{inst.pred}"
    elif inst.op in config.CAST_OPS:
        # A cast is the only op whose result width cannot be read off its operands
        # or its user, so it carries the width in its name.
        head = f"{inst.op}.i{inst.width}"
    text = f"{head}({','.join(parts)})"
    if inst.name in marked:
        text += "!"
    return text


def to_text(fragment: Fragment, mark_external: bool = False) -> str:
    """Print the fragment as one nested expression, such as `add(and(a0,a1),or(a0,a1)):i32`.

    With `mark_external`, an interior instruction that has uses outside the fragment is
    followed by `!`.
    """
    marked: frozenset[str] = frozenset()
    if mark_external:
        marked = frozenset(fragment.external_uses)
    body = arg_text(fragment.root, inst_by_name(fragment), marked)
    if isinstance(fragment.root, Const):
        return body
    return f"{body}:i{root_width(fragment)}"


def input_types_text(fragment: Fragment) -> str:
    """Input widths in order, such as `i32 i32`. Together with to_text this fixes all types."""
    parts = []
    for fragment_input in fragment.inputs:
        parts.append(f"i{fragment_input.width}")
    return " ".join(parts)


def arg_from_json(value: object, context: str) -> Arg:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if set(value.keys()) != {"const", "width"}:
            raise FragmentError(f"{context}: a constant needs exactly 'const' and 'width'")
        text = value["const"]
        width = value["width"]
        if not isinstance(text, str) or not text.isdigit():
            raise FragmentError(f"{context}: 'const' must be an unsigned decimal string")
        if not isinstance(width, int) or isinstance(width, bool):
            raise FragmentError(f"{context}: 'width' must be an integer")
        return Const(int(text), width)
    raise FragmentError(f"{context}: operand must be a name or a constant object")


def arg_to_json(arg: Arg) -> object:
    if isinstance(arg, Const):
        return {"const": str(arg.value), "width": arg.width}
    return arg


def require_field(record: dict, key: str, expected_type: type, context: str) -> object:
    if key not in record:
        raise FragmentError(f"{context}: missing field {key!r}")
    value = record[key]
    if not isinstance(value, expected_type) or isinstance(value, bool):
        raise FragmentError(f"{context}: field {key!r} must be {expected_type.__name__}")
    return value


def location_from_json(record: object) -> Location:
    if not isinstance(record, dict):
        raise FragmentError("location must be an object")
    return Location(
        corpus=require_field(record, "corpus", str, "location"),
        file=require_field(record, "file", str, "location"),
        function=require_field(record, "function", str, "location"),
        block=require_field(record, "block", int, "location"),
    )


def string_tuple(record: dict, key: str) -> tuple[str, ...]:
    values = record.get(key, [])
    if not isinstance(values, list):
        raise FragmentError(f"field {key!r} must be a list")
    for value in values:
        if not isinstance(value, str):
            raise FragmentError(f"field {key!r} must hold strings")
    return tuple(values)


def load_fragment(record: object, harvested: bool = False) -> Fragment:
    """Build a validated Fragment from one decoded JSON object of the fragment contract."""
    if not isinstance(record, dict):
        raise FragmentError("a fragment must be a JSON object")
    version = record.get("version")
    if version != CONTRACT_VERSION:
        raise FragmentError(f"unsupported fragment version {version!r}")

    inputs = []
    for item in require_field(record, "inputs", list, "fragment"):
        if not isinstance(item, dict):
            raise FragmentError("each input must be an object")
        name = require_field(item, "name", str, "input")
        width = require_field(item, "width", int, f"input {name}")
        inputs.append(Input(name, width))

    insts = []
    for item in require_field(record, "insts", list, "fragment"):
        if not isinstance(item, dict):
            raise FragmentError("each instruction must be an object")
        name = require_field(item, "name", str, "instruction")
        op = require_field(item, "op", str, name)
        width = require_field(item, "width", int, name)
        args = []
        for raw_arg in require_field(item, "args", list, name):
            args.append(arg_from_json(raw_arg, name))
        pred = item.get("pred")
        if pred is not None and not isinstance(pred, str):
            raise FragmentError(f"{name}: 'pred' must be a string")
        insts.append(Inst(name, op, width, tuple(args), pred))

    if "root" not in record:
        raise FragmentError("fragment: missing field 'root'")
    root = arg_from_json(record["root"], "root")

    location = None
    if record.get("location") is not None:
        location = location_from_json(record["location"])

    orig_id = record.get("orig_id")
    if orig_id is not None and (not isinstance(orig_id, int) or isinstance(orig_id, bool)):
        raise FragmentError("'orig_id' must be an integer")

    fragment = Fragment(
        inputs=tuple(inputs),
        insts=tuple(insts),
        root=root,
        external_uses=string_tuple(record, "external_uses"),
        stripped_flags=string_tuple(record, "stripped_flags"),
        location=location,
        orig_id=orig_id,
    )
    validate(fragment, harvested=harvested)
    return fragment


def fragment_to_dict(fragment: Fragment) -> dict:
    """Encode a fragment as a JSON-ready object. `load_fragment` reverses this exactly."""
    inputs = []
    for fragment_input in fragment.inputs:
        inputs.append({"name": fragment_input.name, "width": fragment_input.width})

    insts = []
    for inst in fragment.insts:
        args = []
        for arg in inst.args:
            args.append(arg_to_json(arg))
        item = {"name": inst.name, "op": inst.op, "width": inst.width, "args": args}
        if inst.pred is not None:
            item["pred"] = inst.pred
        insts.append(item)

    record: dict = {
        "version": CONTRACT_VERSION,
        "inputs": inputs,
        "insts": insts,
        "root": arg_to_json(fragment.root),
        "external_uses": list(fragment.external_uses),
        "stripped_flags": list(fragment.stripped_flags),
    }
    if fragment.location is not None:
        record["location"] = {
            "corpus": fragment.location.corpus,
            "file": fragment.location.file,
            "function": fragment.location.function,
            "block": fragment.location.block,
        }
    if fragment.orig_id is not None:
        record["orig_id"] = fragment.orig_id
    return record
