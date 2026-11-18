"""Unit tests for the fragment data model: validation, printing, and the JSON contract."""

import json

import pytest
from helpers import build, const, leaf

from superopt.ir import (
    Const,
    Fragment,
    FragmentError,
    Input,
    Inst,
    format_const,
    fragment_to_dict,
    input_types_text,
    load_fragment,
    to_text,
    validate,
)

CONTRACT_EXAMPLE = {
    "version": 1,
    "location": {
        "corpus": "sqlite",
        "file": "sqlite3.c",
        "function": "sqlite3VdbeExec",
        "block": 812,
    },
    "inputs": [{"name": "a0", "width": 32}, {"name": "a1", "width": 32}],
    "insts": [
        {"name": "t0", "op": "and", "width": 32, "args": ["a0", "a1"]},
        {"name": "t1", "op": "or", "width": 32, "args": ["a0", "a1"]},
        {"name": "t2", "op": "add", "width": 32, "args": ["t0", "t1"]},
    ],
    "root": "t2",
    "external_uses": [],
    "stripped_flags": ["nsw"],
}


def test_contract_example_loads_and_prints():
    fragment = load_fragment(CONTRACT_EXAMPLE, harvested=True)
    assert to_text(fragment) == "add(and(a0,a1),or(a0,a1)):i32"
    assert input_types_text(fragment) == "i32 i32"
    assert fragment.location.function == "sqlite3VdbeExec"
    assert fragment.stripped_flags == ("nsw",)


def test_json_round_trip_is_identity():
    fragment = load_fragment(CONTRACT_EXAMPLE, harvested=True)
    encoded = json.dumps(fragment_to_dict(fragment))
    assert load_fragment(json.loads(encoded), harvested=True) == fragment
    assert fragment_to_dict(fragment) == CONTRACT_EXAMPLE


def test_text_marks_external_uses():
    fragment = build(
        [32, 32],
        [("and", 32, "a0", "a1"), ("or", 32, "a0", "a1"), ("add", 32, "t0", "t1")],
        external_uses=("t0",),
    )
    assert to_text(fragment, mark_external=True) == "add(and(a0,a1)!,or(a0,a1)):i32"
    assert to_text(fragment) == "add(and(a0,a1),or(a0,a1)):i32"


def test_text_for_casts_icmp_and_select():
    fragment = build(
        [32, 8],
        [
            ("trunc", 8, "a0"),
            ("icmp.ult", 1, "t0", "a1"),
            ("zext", 32, "a1"),
            ("select", 32, "t1", "t2", const(-1, 32)),
        ],
    )
    expected = "select(icmp.ult(trunc.i8(a0),a1),zext.i32(a1),-1:i32):i32"
    assert to_text(fragment) == expected


def test_format_const():
    assert format_const(Const(0, 32)) == "0:i32"
    assert format_const(Const(255, 8)) == "-1:i8"
    assert format_const(Const(1, 1)) == "1:i1"
    assert format_const(Const(0xFFFFFFFF, 32)) == "-1:i32"
    assert format_const(Const(0x80000000, 32)) == "0x80000000:i32"
    assert format_const(Const(0x1FFFFFFF, 32)) == "0x1fffffff:i32"
    assert format_const(Const(65535, 32)) == "65535:i32"


def test_leaf_fragments_are_valid():
    assert to_text(leaf([32], "a0")) == "a0:i32"
    assert to_text(leaf([32], Const(7, 32))) == "7:i32"


def make(insts: list[Inst], root: str, inputs=None) -> Fragment:
    if inputs is None:
        inputs = (Input("a0", 32), Input("a1", 32))
    return Fragment(inputs=inputs, insts=tuple(insts), root=root)


@pytest.mark.parametrize(
    "fragment",
    [
        # operand used before it is defined
        make([Inst("t0", "add", 32, ("a0", "t1")), Inst("t1", "add", 32, ("a0", "a1"))], "t1"),
        # operand width differs from result width
        make([Inst("t0", "add", 8, ("a0", "a1"))], "t0"),
        # wrong operand count
        make([Inst("t0", "add", 32, ("a0",))], "t0"),
        # icmp without a predicate, and with a bad one
        make([Inst("t0", "icmp", 1, ("a0", "a1"))], "t0"),
        make([Inst("t0", "icmp", 1, ("a0", "a1"), "lt")], "t0"),
        # icmp result must be i1
        make([Inst("t0", "icmp", 32, ("a0", "a1"), "eq")], "t0"),
        # a predicate on a non-icmp
        make([Inst("t0", "add", 32, ("a0", "a1"), "eq")], "t0"),
        # casts in the wrong direction
        make([Inst("t0", "zext", 32, ("a0",))], "t0"),
        make([Inst("t0", "trunc", 32, ("a0",))], "t0"),
        make([Inst("t0", "sext", 8, ("a0",))], "t0"),
        # select condition must be i1
        make([Inst("t0", "select", 32, ("a0", "a0", "a1"))], "t0"),
        # root is not the last instruction
        make([Inst("t0", "add", 32, ("a0", "a1")), Inst("t1", "sub", 32, ("t0", "a1"))], "t0"),
        # dead instruction
        make([Inst("t0", "add", 32, ("a0", "a1")), Inst("t1", "sub", 32, ("a0", "a1"))], "t1"),
        # unknown op
        make([Inst("t0", "udiv", 32, ("a0", "a1"))], "t0"),
        # width that is not allowed
        make([Inst("t0", "add", 24, ("a0", "a1"))], "t0", (Input("a0", 24), Input("a1", 24))),
        # constant out of range
        make([Inst("t0", "add", 32, ("a0", Const(1 << 32, 32)))], "t0"),
        # duplicate name
        make([Inst("a0", "add", 32, ("a0", "a1"))], "a0"),
        # undefined root
        make([], "t9"),
    ],
)
def test_validator_rejects_malformed(fragment):
    with pytest.raises(FragmentError):
        validate(fragment)


def test_harvested_mode_requires_used_inputs_and_an_instruction():
    unused_input = make([Inst("t0", "add", 32, ("a0", "a0"))], "t0")
    validate(unused_input)
    with pytest.raises(FragmentError):
        validate(unused_input, harvested=True)
    with pytest.raises(FragmentError):
        validate(leaf([32], "a0"), harvested=True)


def test_external_use_must_be_an_interior_instruction():
    base = [Inst("t0", "and", 32, ("a0", "a1")), Inst("t1", "add", 32, ("t0", "a1"))]
    validate(Fragment((Input("a0", 32), Input("a1", 32)), tuple(base), "t1", ("t0",)))
    with pytest.raises(FragmentError):
        validate(Fragment((Input("a0", 32), Input("a1", 32)), tuple(base), "t1", ("t1",)))
    with pytest.raises(FragmentError):
        validate(Fragment((Input("a0", 32), Input("a1", 32)), tuple(base), "t1", ("a0",)))


@pytest.mark.parametrize(
    "mutation",
    [
        {"version": 2},
        {"root": 5},
        {"inputs": "a0"},
        {"insts": [{"name": "t0", "op": "add", "width": "32", "args": ["a0", "a1"]}]},
        {"insts": [{"name": "t0", "op": "add", "width": 32, "args": [{"const": "-1"}]}]},
        {
            "insts": [
                {
                    "name": "t0",
                    "op": "add",
                    "width": 32,
                    "args": ["a0", {"const": "-1", "width": 32}],
                }
            ],
            "root": "t0",
        },
        {"stripped_flags": ["fast"]},
        {"external_uses": "t0"},
    ],
)
def test_loader_rejects_bad_json(mutation):
    record = dict(CONTRACT_EXAMPLE)
    record.update(mutation)
    with pytest.raises(FragmentError):
        load_fragment(record)


def test_loader_rejects_missing_fields():
    for key in ("inputs", "insts", "root"):
        record = dict(CONTRACT_EXAMPLE)
        del record[key]
        with pytest.raises(FragmentError):
            load_fragment(record)
