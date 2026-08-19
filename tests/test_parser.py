"""Tests for ir.parser."""

import pytest

from cppl.ir.errors import ParseError
from cppl.ir.models import (
    BinaryOp,
    ConstantOp,
    ExtractOp,
    InstanceOp,
    MemOp,
    Module,
    OutputOp,
    PortDir,
    RegOp,
    UnaryOp,
    VariadicOp,
)
from cppl.ir.parser import parse_design


class TestParseDesign:
    def test_collects_errors_across_modules_and_operations(self):
        raw = [
            {
                "name": "First",
                "ports": {
                    "1": {"dir": "input", "width": 1},
                    "bad_width": {"dir": "output", "width": 0},
                },
                "body": [
                    {"id": "sum", "op": "add", "args": ["a"]},
                    {"op": "unknown"},
                ],
            },
            {
                "name": "Second",
                "ports": {"x": {"dir": "sideways", "width": 1}},
                "body": "not-an-array",
            },
        ]
        with pytest.raises(ParseError) as error:
            parse_design(raw)

        assert len(error.value.issues) == 6
        message = str(error.value)
        assert "port '1'" in message
        assert "bad_width" in message
        assert "exactly 2" in message
        assert "unknown op" in message
        assert "sideways" in message
        assert "body" in message

    def test_single_module_dict(self):
        """A single module dict (not wrapped in array) should work."""
        raw = '{"name":"A","ports":{"x":{"dir":"input","width":1}},"body":[{"op":"output","args":{}}]}'
        modules = parse_design(raw)
        assert len(modules) == 1
        assert modules[0].name == "A"

    def test_array_of_modules(self):
        raw = '[{"name":"A","ports":{"x":{"dir":"input","width":1}},"body":[{"op":"output","args":{}}]}]'
        modules = parse_design(raw)
        assert len(modules) == 1

    def test_invalid_json(self):
        with pytest.raises(ParseError, match="Invalid JSON"):
            parse_design("{bad json")

    def test_missing_name(self):
        with pytest.raises(ParseError, match="name"):
            parse_design('{"ports":{},"body":[]}')

    def test_missing_ports(self):
        with pytest.raises(ParseError, match="ports"):
            parse_design('{"name":"A","body":[]}')

    def test_invalid_port_dir(self):
        with pytest.raises(ParseError, match="dir"):
            parse_design(
                '{"name":"A","ports":{"x":{"dir":"inout","width":1}},"body":[]}'
            )

    def test_invalid_port_width(self):
        with pytest.raises(ParseError, match="width"):
            parse_design(
                '{"name":"A","ports":{"x":{"dir":"input","width":0}},"body":[]}'
            )

    def test_numeric_port_name_is_rejected(self):
        with pytest.raises(ParseError, match="SSA identifier"):
            parse_design(
                '{"name":"A","ports":{"1":{"dir":"input","width":1}},'
                '"body":[{"op":"output","args":{}}]}'
            )

    def test_unknown_op(self):
        raw = '{"name":"A","ports":{"x":{"dir":"input","width":1}},"body":[{"op":"foobar"}]}'
        with pytest.raises(ParseError, match="unknown op"):
            parse_design(raw)


class TestParseOperations:
    def parse_test_module(self, body_json: str) -> Module:
        raw = f'{{"name":"M","ports":{{"a":{{"dir":"input","width":8}},"out":{{"dir":"output","width":8}}}},"body":[{body_json}]}}'
        return parse_design(raw)[0]

    def test_constant(self):
        mod = self.parse_test_module(
            '{"id":"c","op":"constant","value":42,"width":8},{"op":"output","args":{"out":"c"}}'
        )
        assert isinstance(mod.body[0], ConstantOp)
        assert mod.body[0].value == 42
        assert mod.body[0].width == 8

    def test_numeric_ssa_id_is_rejected(self):
        with pytest.raises(ParseError, match="SSA identifier"):
            self.parse_test_module(
                '{"id":"300","op":"constant","value":0,"width":8},'
                '{"op":"output","args":{"out":"300"}}'
            )

    def test_ssa_id_must_start_with_letter_or_underscore(self):
        with pytest.raises(ParseError, match="SSA identifier"):
            self.parse_test_module(
                '{"id":"9value","op":"constant","value":0,"width":8},'
                '{"op":"output","args":{"out":"9value"}}'
            )

    def test_constant_hex_string(self):
        mod = self.parse_test_module(
            '{"id":"c","op":"constant","value":"0xff","width":8},{"op":"output","args":{"out":"c"}}'
        )
        assert isinstance(mod.body[0], ConstantOp)
        assert mod.body[0].value == "0xff"

    def test_unary_not(self):
        mod = self.parse_test_module(
            '{"id":"n","op":"not","args":["a"]},{"op":"output","args":{"out":"n"}}'
        )
        op = mod.body[0]
        assert isinstance(op, UnaryOp)
        assert op.op == "not"
        assert op.args == ["a"]

    def test_binary_add(self):
        mod = self.parse_test_module(
            '{"id":"s","op":"add","args":["a","a"]},{"op":"output","args":{"out":"s"}}'
        )
        op = mod.body[0]
        assert isinstance(op, BinaryOp)
        assert op.args == ["a", "a"]

    def test_binary_wrong_arg_count(self):
        with pytest.raises(ParseError, match="exactly 2"):
            self.parse_test_module(
                '{"id":"s","op":"add","args":["a"]},{"op":"output","args":{"out":"s"}}'
            )

    def test_variadic_concat(self):
        mod = self.parse_test_module(
            '{"id":"c","op":"concat","args":["a","a"]},{"op":"output","args":{"out":"c"}}'
        )
        op = mod.body[0]
        assert isinstance(op, VariadicOp)
        assert op.args == ["a", "a"]

    def test_extract(self):
        mod = self.parse_test_module(
            '{"id":"e","op":"extract","args":["a"],"lowBit":0,"width":4},{"op":"output","args":{"out":"e"}}'
        )
        op = mod.body[0]
        assert isinstance(op, ExtractOp)
        assert op.lowBit == 0
        assert op.width == 4

    def test_instance(self):
        raw = """[
          {"name":"Sub","ports":{"i":{"dir":"input","width":8},"o":{"dir":"output","width":8}},
           "body":[{"op":"output","args":{"o":"i"}}]},
          {"name":"Top","ports":{"a":{"dir":"input","width":8},"out":{"dir":"output","width":8}},
           "body":[
             {"id":["r"],"op":"instance","module":"Sub","args":{"i":"a"}},
             {"op":"output","args":{"out":"r"}}
           ]}
        ]"""
        modules = parse_design(raw)
        inst = modules[1].body[0]
        assert isinstance(inst, InstanceOp)
        assert inst.module == "Sub"
        assert inst.id == ["r"]

    def test_numeric_instance_output_id_is_rejected(self):
        raw = """[
          {"name":"Sub","ports":{"i":{"dir":"input","width":8},"o":{"dir":"output","width":8}},
           "body":[{"op":"output","args":{"o":"i"}}]},
          {"name":"Top","ports":{"a":{"dir":"input","width":8},"out":{"dir":"output","width":8}},
           "body":[
             {"id":["1"],"op":"instance","module":"Sub","args":{"i":"a"}},
             {"op":"output","args":{"out":"1"}}
           ]}
        ]"""
        with pytest.raises(ParseError, match="SSA identifier"):
            parse_design(raw)

    def test_output(self):
        mod = self.parse_test_module('{"op":"output","args":{"out":"a"}}')
        op = mod.body[0]
        assert isinstance(op, OutputOp)
        assert op.args == {"out": "a"}

    def test_reg_basic(self):
        raw = """{
          "name":"M","ports":{"clk":{"dir":"input","width":1},"d":{"dir":"input","width":8},"q":{"dir":"output","width":8}},
          "body":[
            {"id":"r","op":"reg","args":["d"],"clock":"clk"},
            {"op":"output","args":{"q":"r"}}
          ]
        }"""
        mod = parse_design(raw)[0]
        op = mod.body[0]
        assert isinstance(op, RegOp)
        assert op.id == "r"
        assert op.args == ["d"]
        assert op.clock == "clk"
        assert op.reset == ""
        assert op.enable == ""

    def test_reg_with_reset(self):
        raw = """{
          "name":"M","ports":{"clk":{"dir":"input","width":1},"rst":{"dir":"input","width":1},"d":{"dir":"input","width":8},"q":{"dir":"output","width":8}},
          "body":[
            {"id":"r","op":"reg","args":["d"],"clock":"clk","reset":"rst","resetValue":0},
            {"op":"output","args":{"q":"r"}}
          ]
        }"""
        mod = parse_design(raw)[0]
        op = mod.body[0]
        assert isinstance(op, RegOp)
        assert op.reset == "rst"
        assert op.resetValue == 0

    def test_reg_with_enable(self):
        raw = """{
          "name":"M","ports":{"clk":{"dir":"input","width":1},"en":{"dir":"input","width":1},"d":{"dir":"input","width":8},"q":{"dir":"output","width":8}},
          "body":[
            {"id":"r","op":"reg","args":["d"],"clock":"clk","enable":"en"},
            {"op":"output","args":{"q":"r"}}
          ]
        }"""
        mod = parse_design(raw)[0]
        op = mod.body[0]
        assert isinstance(op, RegOp)
        assert op.enable == "en"

    def test_reg_full(self):
        raw = """{
          "name":"M","ports":{"clk":{"dir":"input","width":1},"rst":{"dir":"input","width":1},"en":{"dir":"input","width":1},"d":{"dir":"input","width":8},"q":{"dir":"output","width":8}},
          "body":[
            {"id":"r","op":"reg","args":["d"],"clock":"clk","reset":"rst","resetValue":0,"enable":"en"},
            {"op":"output","args":{"q":"r"}}
          ]
        }"""
        mod = parse_design(raw)[0]
        op = mod.body[0]
        assert isinstance(op, RegOp)
        assert op.clock == "clk"
        assert op.reset == "rst"
        assert op.resetValue == 0
        assert op.enable == "en"

    def test_reg_missing_clock(self):
        with pytest.raises(ParseError, match="clock"):
            parse_design(
                """{
              "name":"M","ports":{"d":{"dir":"input","width":8},"q":{"dir":"output","width":8}},
              "body":[
                {"id":"r","op":"reg","args":["d"]},
                {"op":"output","args":{"q":"r"}}
              ]
            }"""
            )

    def test_reg_reset_without_resetValue(self):
        with pytest.raises(ParseError, match="resetValue.*required"):
            parse_design(
                """{
              "name":"M","ports":{"clk":{"dir":"input","width":1},"rst":{"dir":"input","width":1},"d":{"dir":"input","width":8},"q":{"dir":"output","width":8}},
              "body":[
                {"id":"r","op":"reg","args":["d"],"clock":"clk","reset":"rst"},
                {"op":"output","args":{"q":"r"}}
              ]
            }"""
            )


class TestParseMem:
    def test_numeric_memory_output_id_is_rejected(self):
        raw = """{
          "name":"M",
          "ports":{"clk":{"dir":"input","width":1},"o":{"dir":"output","width":8}},
          "body":[
            {"id":["1"],"op":"mem","width":8,"depth":1,"clock":"clk",
             "reads":[{"addr":"0","enable":"1"}],"writes":[]},
            {"op":"output","args":{"o":"1"}}
          ]
        }"""
        with pytest.raises(ParseError, match="SSA identifier"):
            parse_design(raw)

    def test_mem_op(self):
        raw = """{
          "name":"M",
          "ports":{
            "clk":{"dir":"input","width":1},
            "rst":{"dir":"input","width":1},
            "addr":{"dir":"input","width":5},
            "wdata":{"dir":"input","width":32},
            "wen":{"dir":"input","width":1},
            "ren":{"dir":"input","width":1},
            "rdata":{"dir":"output","width":32}
          },
          "body":[
            {"id":["rd"],"op":"mem","width":32,"depth":32,
             "clock":"clk","reset":"rst",
             "reads":[{"addr":"addr","enable":"ren"}],
             "writes":[{"addr":"addr","data":"wdata","enable":"wen"}]},
            {"op":"output","args":{"rdata":"rd"}}
          ]
        }"""
        mod = parse_design(raw)[0]
        op = mod.body[0]
        assert isinstance(op, MemOp)
        assert op.id == ["rd"]
        assert op.width == 32
        assert op.depth == 32
        assert op.clock == "clk"
        assert op.reset == "rst"
        assert op.reads == (("addr", "ren"),)
        assert op.writes == (("addr", "wdata", "wen", ""),)

    def test_mem_id_reads_mismatch(self):
        raw = """{
          "name":"M",
          "ports":{
            "clk":{"dir":"input","width":1},"rst":{"dir":"input","width":1},
            "addr":{"dir":"input","width":5},"ren":{"dir":"input","width":1},
            "rdata":{"dir":"output","width":32}
          },
          "body":[
            {"id":["rd1","rd2"],"op":"mem","width":32,"depth":32,
             "clock":"clk","reset":"rst",
             "reads":[{"addr":"addr","enable":"ren"}],
             "writes":[]},
            {"op":"output","args":{"rdata":"rd1"}}
          ]
        }"""
        with pytest.raises(ParseError, match="one output id per read port"):
            parse_design(raw)

    def test_mem_missing_clock(self):
        raw = """{
          "name":"M",
          "ports":{"rst":{"dir":"input","width":1},"o":{"dir":"output","width":32}},
          "body":[
            {"id":["rd"],"op":"mem","width":32,"depth":32,
             "reset":"rst","reads":[{"addr":"x","enable":"y"}],"writes":[]},
            {"op":"output","args":{"o":"rd"}}
          ]
        }"""
        with pytest.raises(ParseError, match="clock"):
            parse_design(raw)
