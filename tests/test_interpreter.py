"""Tests for the pure-Python CPPL IR interpreter."""

import json

import pytest

from cppl import Interpreter, SimulationError
from cppl.design import Design
from cppl.ir.infer import infer_widths
from cppl.ir.parser import parse_design


def build_single_operation_simulator(
    op, *, width=8, output_width=None, extra_ports=None
):
    output_width = width if output_width is None else output_width
    ports = {
        "a": {"dir": "input", "width": width},
        "b": {"dir": "input", "width": width},
        "o": {"dir": "output", "width": output_width},
    }
    ports.update(extra_ports or {})
    return Interpreter.from_json(
        [
            {
                "name": "Top",
                "ports": ports,
                "body": [op, {"op": "output", "args": {"o": op["id"]}}],
            }
        ]
    )


class TestCombinational:
    @pytest.mark.parametrize(
        ("op", "a", "b", "expected"),
        [
            ("add", 250, 10, 4),
            ("sub", 3, 5, 254),
            ("mul", 20, 20, 144),
            ("div", 255, 2, 127),
            ("div_s", 0xF9, 2, 0xFD),
            ("mod_u", 13, 5, 3),
            ("mod_s", 0xF9, 3, 0xFF),
            ("and", 0xA5, 0x3C, 0x24),
            ("or", 0xA5, 0x3C, 0xBD),
            ("xor", 0xA5, 0x3C, 0x99),
            ("shl", 3, 2, 12),
            ("shr_u", 0x80, 2, 0x20),
            ("shr_s", 0x80, 2, 0xE0),
        ],
    )
    def test_binary_ops(self, op, a, b, expected):
        sim = build_single_operation_simulator(
            {"id": "r", "op": op, "args": ["a", "b"]}
        )
        assert sim.evaluate({"a": a, "b": b}) == {"o": expected}

    @pytest.mark.parametrize(
        ("op", "a", "b", "expected"),
        [
            ("eq", 4, 4, 1),
            ("ne", 4, 5, 1),
            ("lt_u", 1, 255, 1),
            ("ge_u", 255, 1, 1),
            ("gt_u", 5, 4, 1),
            ("le_u", 4, 4, 1),
            ("lt_s", 0xFF, 1, 1),
            ("ge_s", 0xFF, 1, 0),
            ("gt_s", 1, 0xFF, 1),
            ("le_s", 0xFF, 0xFF, 1),
        ],
    )
    def test_compare_ops(self, op, a, b, expected):
        sim = build_single_operation_simulator(
            {"id": "r", "op": op, "args": ["a", "b"]}, output_width=1
        )
        assert sim.evaluate({"a": a, "b": b}) == {"o": expected}

    @pytest.mark.parametrize(
        ("op", "value", "expected", "output_width"),
        [
            ("not", 0x0F, 0xF0, 8),
            ("neg", 2, 0xFE, 8),
            ("reverse", 0b00010110, 0b01101000, 8),
            ("or_reduce", 0x10, 1, 1),
            ("and_reduce", 0xFF, 1, 1),
            ("xor_reduce", 0b1011, 1, 1),
        ],
    )
    def test_unary_ops(self, op, value, expected, output_width):
        sim = build_single_operation_simulator(
            {"id": "r", "op": op, "args": ["a"]},
            output_width=output_width,
        )
        assert sim.evaluate({"a": value}) == {"o": expected}

    def test_concat_extract_mux_and_casts(self):
        raw = [
            {
                "name": "Top",
                "ports": {
                    "a": {"dir": "input", "width": 4},
                    "b": {"dir": "input", "width": 4},
                    "sel": {"dir": "input", "width": 1},
                    "concat": {"dir": "output", "width": 8},
                    "extract": {"dir": "output", "width": 3},
                    "muxed": {"dir": "output", "width": 4},
                    "sext": {"dir": "output", "width": 8},
                    "zext": {"dir": "output", "width": 8},
                },
                "body": [
                    {"id": "c", "op": "concat", "args": ["a", "b"]},
                    {
                        "id": "e",
                        "op": "extract",
                        "args": ["c"],
                        "lowBit": 2,
                        "width": 3,
                    },
                    {"id": "m", "op": "mux", "args": ["sel", "a", "b"]},
                    {"id": "s", "op": "sext", "args": ["a"], "width": 8},
                    {"id": "z", "op": "zext", "args": ["a"], "width": 8},
                    {
                        "op": "output",
                        "args": {
                            "concat": "c",
                            "extract": "e",
                            "muxed": "m",
                            "sext": "s",
                            "zext": "z",
                        },
                    },
                ],
            }
        ]
        sim = Interpreter.from_json(raw)
        assert sim.evaluate({"a": 0xA, "b": 0x3, "sel": 1}) == {
            "concat": 0xA3,
            "extract": 0,
            "muxed": 0xA,
            "sext": 0xFA,
            "zext": 0x0A,
        }

    def test_forward_reference_and_input_truncation(self):
        raw = [
            {
                "name": "Top",
                "ports": {
                    "a": {"dir": "input", "width": 8},
                    "o": {"dir": "output", "width": 8},
                },
                "body": [
                    {"id": "sum", "op": "add", "args": ["a", "one"]},
                    {"id": "one", "op": "constant", "value": 1, "width": 8},
                    {"op": "output", "args": {"o": "sum"}},
                ],
            }
        ]
        sim = Interpreter.from_json(raw)
        assert sim.evaluate({"a": 0x1FF}) == {"o": 0}
        assert sim.peek("Top.sum") == 0

    @pytest.mark.parametrize("op", ["div", "div_s", "mod_u", "mod_s"])
    def test_division_by_zero(self, op):
        sim = build_single_operation_simulator(
            {"id": "r", "op": op, "args": ["a", "b"]}
        )
        with pytest.raises(SimulationError, match="Division by zero"):
            sim.evaluate({"a": 1, "b": 0})


REG_JSON = [
    {
        "name": "Reg",
        "ports": {
            "clk": {"dir": "input", "width": 1, "type": "clock"},
            "rst": {"dir": "input", "width": 1},
            "en": {"dir": "input", "width": 1},
            "d": {"dir": "input", "width": 8},
            "q": {"dir": "output", "width": 8},
        },
        "body": [
            {
                "id": "q_reg",
                "op": "reg",
                "args": ["d"],
                "clock": "clk",
                "reset": "rst",
                "resetValue": "0x5a",
                "enable": "en",
            },
            {"op": "output", "args": {"q": "q_reg"}},
        ],
    }
]


class TestRegisters:
    def test_edges_reset_enable_and_reset_state(self):
        sim = Interpreter.from_json(REG_JSON)
        assert sim.peek_outputs() == {"q": 0}
        assert sim.evaluate({"clk": 0, "d": 9, "en": 1}) == {"q": 0}
        assert sim.evaluate({"clk": 1}) == {"q": 9}
        assert sim.evaluate({"d": 12}) == {"q": 9}
        assert sim.evaluate({"clk": 0, "en": 0}) == {"q": 9}
        assert sim.evaluate({"clk": 1, "d": 13}) == {"q": 9}
        assert sim.evaluate({"clk": 0, "rst": 1}) == {"q": 9}
        assert sim.evaluate({"clk": 1}) == {"q": 0x5A}
        sim.reset_state()
        assert sim.peek_outputs() == {"q": 0}

    def test_two_clock_domains_update_independently(self):
        raw = [
            {
                "name": "Top",
                "ports": {
                    "c1": {"dir": "input", "width": 1},
                    "c2": {"dir": "input", "width": 1},
                    "d1": {"dir": "input", "width": 8},
                    "d2": {"dir": "input", "width": 8},
                    "q1": {"dir": "output", "width": 8},
                    "q2": {"dir": "output", "width": 8},
                },
                "body": [
                    {"id": "r1", "op": "reg", "args": ["d1"], "clock": "c1"},
                    {"id": "r2", "op": "reg", "args": ["d2"], "clock": "c2"},
                    {"op": "output", "args": {"q1": "r1", "q2": "r2"}},
                ],
            }
        ]
        sim = Interpreter.from_json(raw)
        assert sim.evaluate({"d1": 1, "d2": 2, "c1": 1}) == {"q1": 1, "q2": 0}
        assert sim.evaluate({"c1": 0, "c2": 1}) == {"q1": 1, "q2": 2}


MEM_JSON = [
    {
        "name": "MemTop",
        "ports": {
            "clk": {"dir": "input", "width": 1},
            "rst": {"dir": "input", "width": 1},
            "raddr": {"dir": "input", "width": 2},
            "waddr": {"dir": "input", "width": 2},
            "wdata": {"dir": "input", "width": 8},
            "ren": {"dir": "input", "width": 1},
            "wen": {"dir": "input", "width": 1},
            "rdata": {"dir": "output", "width": 8},
        },
        "body": [
            {
                "id": ["rd"],
                "op": "mem",
                "width": 8,
                "depth": 4,
                "clock": "clk",
                "reset": "rst",
                "name": "storage",
                "reads": [{"addr": "raddr", "enable": "ren"}],
                "writes": [{"addr": "waddr", "data": "wdata", "enable": "wen"}],
            },
            {"op": "output", "args": {"rdata": "rd"}},
        ],
    }
]


class TestMemory:
    def test_combinational_read_synchronous_write_and_reset(self):
        sim = Interpreter.from_json(MEM_JSON)
        assert sim.evaluate(
            {
                "clk": 0,
                "raddr": 2,
                "waddr": 2,
                "wdata": 0xAB,
                "ren": 1,
                "wen": 1,
            }
        ) == {"rdata": 0}
        assert sim.evaluate({"clk": 1}) == {"rdata": 0xAB}
        assert sim.peek_memory("storage") == (0, 0, 0xAB, 0)
        assert sim.evaluate({"ren": 0}) == {"rdata": 0}
        sim.evaluate({"clk": 0, "rst": 1})
        assert sim.evaluate({"clk": 1, "ren": 1}) == {"rdata": 0}
        assert sim.peek_memory("MemTop.storage") == (0, 0, 0, 0)

    def test_masked_write_preserves_unselected_bits(self):
        raw = json.loads(json.dumps(MEM_JSON))
        raw[0]["ports"]["wmask"] = {"dir": "input", "width": 8}
        raw[0]["body"][0]["writes"][0]["mask"] = "wmask"
        sim = Interpreter.from_json(raw)

        sim.evaluate(
            {
                "clk": 0,
                "wen": 1,
                "waddr": 2,
                "wdata": 0xAB,
                "wmask": 0x0F,
            }
        )
        sim.evaluate({"clk": 1})
        assert sim.peek_memory("storage")[2] == 0x0B

        sim.evaluate({"clk": 0, "wdata": 0xC0, "wmask": 0xF0})
        sim.evaluate({"clk": 1})
        assert sim.peek_memory("storage")[2] == 0xCB

    def test_hierarchical_multi_read_memory_resolves_ports_independently(self):
        raw = [
            {
                "name": "DualMem",
                "ports": {
                    "clk": {"dir": "input", "width": 1},
                    "a": {"dir": "input", "width": 2},
                    "b": {"dir": "input", "width": 2},
                    "first": {"dir": "output", "width": 2},
                    "second": {"dir": "output", "width": 2},
                },
                "body": [
                    {"id": "one", "op": "constant", "value": 1, "width": 1},
                    {
                        "id": ["first_value", "second_value"],
                        "op": "mem",
                        "width": 2,
                        "depth": 4,
                        "clock": "clk",
                        "name": "mem",
                        "reads": [
                            {"addr": "a", "enable": "one"},
                            {"addr": "b", "enable": "one"},
                        ],
                        "writes": [],
                    },
                    {
                        "op": "output",
                        "args": {"first": "first_value", "second": "second_value"},
                    },
                ],
            },
            {
                "name": "Top",
                "ports": {
                    "clk": {"dir": "input", "width": 1},
                    "a": {"dir": "input", "width": 2},
                    "out": {"dir": "output", "width": 2},
                },
                "body": [
                    {
                        "id": ["first", "second"],
                        "op": "instance",
                        "module": "DualMem",
                        "name": "u",
                        "args": {"clk": "clk", "a": "a", "b": "derived_b"},
                    },
                    {
                        "id": "derived_b",
                        "op": "extract",
                        "args": ["first"],
                        "lowBit": 0,
                        "width": 2,
                    },
                    {"op": "output", "args": {"out": "second"}},
                ],
            },
        ]
        modules = parse_design(raw)
        sim = Interpreter(
            modules,
            top="Top",
            memory_fixtures={"u.mem": {1: 2, 2: 3}},
        )
        assert sim.evaluate({"clk": 0, "a": 1}) == {"out": 3}

    def test_sequential_child_receives_late_resolved_inputs(self):
        raw = [
            {
                "name": "State",
                "ports": {
                    "clk": {"dir": "input", "width": 1},
                    "d": {"dir": "input", "width": 8},
                    "en": {"dir": "input", "width": 1},
                    "q": {"dir": "output", "width": 8},
                },
                "body": [
                    {
                        "id": "state",
                        "op": "reg",
                        "args": ["d"],
                        "clock": "clk",
                        "enable": "en",
                        "width": 8,
                    },
                    {"op": "output", "args": {"q": "state"}},
                ],
            },
            {
                "name": "Top",
                "ports": {
                    "clk": {"dir": "input", "width": 1},
                    "en": {"dir": "input", "width": 1},
                    "q": {"dir": "output", "width": 8},
                },
                "body": [
                    {"id": "one", "op": "constant", "value": 1, "width": 8},
                    {
                        "id": ["child_q"],
                        "op": "instance",
                        "module": "State",
                        "name": "state0",
                        "args": {"clk": "clk", "d": "next_q", "en": "en"},
                    },
                    {"id": "next_q", "op": "add", "args": ["child_q", "one"]},
                    {"op": "output", "args": {"q": "child_q"}},
                ],
            },
        ]
        sim = Interpreter.from_json(raw, top="Top")
        assert sim.evaluate({"clk": 0, "en": 1}) == {"q": 0}
        assert sim.evaluate({"clk": 1}) == {"q": 1}

    def test_init_file(self, tmp_path):
        init_file = tmp_path / "memory.hex"
        init_file.write_text("01 02\n@3 ff // comment\n")
        raw = json.loads(json.dumps(MEM_JSON))
        raw[0]["body"][0]["initFile"] = "memory.hex"
        sim = Interpreter.from_json(raw, base_dir=tmp_path)
        assert sim.peek_memory("storage") == (1, 2, 0, 0xFF)
        assert sim.evaluate({"ren": 1, "raddr": 3}) == {"rdata": 0xFF}

    def test_missing_init_file(self, tmp_path):
        raw = json.loads(json.dumps(MEM_JSON))
        raw[0]["body"][0]["initFile"] = "missing.hex"
        with pytest.raises(SimulationError, match="cannot read initFile"):
            Interpreter.from_json(raw, base_dir=tmp_path)

    def test_enabled_out_of_range_address(self):
        raw = json.loads(json.dumps(MEM_JSON))
        raw[0]["ports"]["raddr"]["width"] = 3
        raw[0]["ports"]["waddr"]["width"] = 3
        # Width inference deliberately rejects this before simulation.  Use a
        # non-power-of-two memory to make an in-width address out of range.
        raw[0]["body"][0]["depth"] = 5
        sim = Interpreter.from_json(raw)
        with pytest.raises(SimulationError, match="outside depth"):
            sim.evaluate({"ren": 1, "raddr": 7})

    def test_conflicting_writes(self):
        raw = json.loads(json.dumps(MEM_JSON))
        raw[0]["body"][0]["writes"].append(
            {"addr": "waddr", "data": "wdata", "enable": "wen"}
        )
        sim = Interpreter.from_json(raw)
        sim.evaluate({"clk": 0, "wen": 1, "waddr": 1, "wdata": 3})
        with pytest.raises(SimulationError, match="Conflicting writes"):
            sim.evaluate({"clk": 1})


class TestHierarchyAndAPI:
    def test_instances_have_isolated_state_and_are_observable(self):
        raw = [
            {
                "name": "Child",
                "ports": {
                    "clk": {"dir": "input", "width": 1},
                    "d": {"dir": "input", "width": 8},
                    "q": {"dir": "output", "width": 8},
                },
                "body": [
                    {"id": "r", "op": "reg", "args": ["d"], "clock": "clk"},
                    {"op": "output", "args": {"q": "r"}},
                ],
            },
            {
                "name": "Top",
                "ports": {
                    "clk": {"dir": "input", "width": 1},
                    "a": {"dir": "input", "width": 8},
                    "b": {"dir": "input", "width": 8},
                    "qa": {"dir": "output", "width": 8},
                    "qb": {"dir": "output", "width": 8},
                },
                "body": [
                    {
                        "id": ["x"],
                        "op": "instance",
                        "module": "Child",
                        "name": "left",
                        "args": {"clk": "clk", "d": "a"},
                    },
                    {
                        "id": ["y"],
                        "op": "instance",
                        "module": "Child",
                        "args": {"clk": "clk", "d": "b"},
                    },
                    {"op": "output", "args": {"qa": "x", "qb": "y"}},
                ],
            },
        ]
        sim = Interpreter.from_json(raw)
        assert sim.top == "Top"
        assert sim.evaluate({"clk": 1, "a": 4, "b": 9}) == {"qa": 4, "qb": 9}
        assert sim.peek("left.r") == 4
        assert sim.peek("Top.child_0.r") == 9

    def test_multiple_roots_require_top(self):
        raw = [
            {
                "name": "A",
                "ports": {"o": {"dir": "output", "width": 1}},
                "body": [
                    {"id": "z", "op": "constant", "value": 0, "width": 1},
                    {"op": "output", "args": {"o": "z"}},
                ],
            },
            {
                "name": "B",
                "ports": {"o": {"dir": "output", "width": 1}},
                "body": [
                    {"id": "z", "op": "constant", "value": 0, "width": 1},
                    {"op": "output", "args": {"o": "z"}},
                ],
            },
        ]
        with pytest.raises(SimulationError, match="multiple root modules"):
            Interpreter.from_json(raw)
        assert Interpreter.from_json(raw, top="B").top == "B"

    def test_instances_settle_across_parent_forward_dependencies(self):
        raw = [
            {
                "name": "Producer",
                "ports": {
                    "feedback": {"dir": "input", "width": 8},
                    "independent": {"dir": "output", "width": 8},
                    "dependent": {"dir": "output", "width": 8},
                },
                "body": [
                    {"id": "one", "op": "constant", "value": 1, "width": 8},
                    {"id": "dep", "op": "add", "args": ["feedback", "one"]},
                    {
                        "op": "output",
                        "args": {
                            "independent": "one",
                            "dependent": "dep",
                        },
                    },
                ],
            },
            {
                "name": "Consumer",
                "ports": {
                    "value": {"dir": "input", "width": 8},
                    "feedback": {"dir": "output", "width": 8},
                },
                "body": [{"op": "output", "args": {"feedback": "value"}}],
            },
            {
                "name": "Top",
                "ports": {"o": {"dir": "output", "width": 8}},
                "body": [
                    {
                        "id": ["first", "result"],
                        "op": "instance",
                        "module": "Producer",
                        "args": {"feedback": "loopback"},
                    },
                    {
                        "id": ["loopback"],
                        "op": "instance",
                        "module": "Consumer",
                        "args": {"value": "first"},
                    },
                    {"op": "output", "args": {"o": "result"}},
                ],
            },
        ]
        sim = Interpreter.from_json(raw)
        assert sim.evaluate() == {"o": 2}

    def test_design_interpreter_uses_ir_pipeline(self, monkeypatch):
        modules = parse_design(REG_JSON)
        widths = infer_widths(modules)
        design = Design()
        monkeypatch.setattr(
            design, "run_ir_pipeline", lambda max_retries=3: (modules, widths)
        )
        sim = design.interpreter(top="Reg")
        assert isinstance(sim, Interpreter)

    def test_invalid_input_and_peek_path(self):
        sim = Interpreter.from_json(REG_JSON)
        with pytest.raises(SimulationError, match="no input port"):
            sim.evaluate({"missing": 1})
        with pytest.raises(SimulationError, match="Unknown value path"):
            sim.peek("missing")
