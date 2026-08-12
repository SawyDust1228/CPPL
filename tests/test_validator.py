"""Tests for ir.validator."""

import pytest

from cppl.ir.errors import CycleError, SSAError, ValidationError
from cppl.ir.parser import parse_design
from cppl.ir.validator import validate_design


def _validate(json_str: str) -> None:
    modules = parse_design(json_str)
    validate_design(modules)


class TestModuleLevel:
    def test_duplicate_module_name(self):
        raw = '''[
          {"name":"A","ports":{"x":{"dir":"input","width":1}},"body":[{"op":"output","args":{}}]},
          {"name":"A","ports":{"y":{"dir":"input","width":1}},"body":[{"op":"output","args":{}}]}
        ]'''
        with pytest.raises(ValidationError, match="Duplicate module name"):
            _validate(raw)

    def test_empty_body(self):
        raw = '{"name":"A","ports":{"x":{"dir":"input","width":1}},"body":[]}'
        with pytest.raises(ValidationError, match="body must not be empty"):
            _validate(raw)

    def test_no_output_terminator(self):
        raw = '{"name":"A","ports":{"x":{"dir":"input","width":8}},"body":[{"id":"c","op":"constant","value":0,"width":8}]}'
        with pytest.raises(ValidationError, match="last operation must be 'output'"):
            _validate(raw)

    def test_output_not_last(self):
        raw = '''{
          "name":"A",
          "ports":{"x":{"dir":"input","width":8},"y":{"dir":"output","width":8}},
          "body":[
            {"op":"output","args":{"y":"x"}},
            {"id":"c","op":"constant","value":0,"width":8}
          ]
        }'''
        with pytest.raises(ValidationError, match="output.*must only appear as the last"):
            _validate(raw)


class TestSSA:
    def test_forward_reference(self):
        raw = '''{
          "name":"A",
          "ports":{"x":{"dir":"input","width":8},"y":{"dir":"output","width":8}},
          "body":[
            {"id":"r","op":"add","args":["x","undefined_val"]},
            {"op":"output","args":{"y":"r"}}
          ]
        }'''
        with pytest.raises(SSAError, match="undefined value 'undefined_val'"):
            _validate(raw)

    def test_shadowing(self):
        raw = '''{
          "name":"A",
          "ports":{"x":{"dir":"input","width":8},"y":{"dir":"output","width":8}},
          "body":[
            {"id":"x","op":"constant","value":0,"width":8},
            {"op":"output","args":{"y":"x"}}
          ]
        }'''
        with pytest.raises(SSAError, match="already defined.*shadowing"):
            _validate(raw)

    def test_id_reuse(self):
        raw = '''{
          "name":"A",
          "ports":{"a":{"dir":"input","width":8},"y":{"dir":"output","width":8}},
          "body":[
            {"id":"r","op":"constant","value":0,"width":8},
            {"id":"r","op":"constant","value":1,"width":8},
            {"op":"output","args":{"y":"r"}}
          ]
        }'''
        with pytest.raises(SSAError, match="already defined"):
            _validate(raw)


class TestOutputCoverage:
    def test_missing_output_port(self):
        raw = '''{
          "name":"A",
          "ports":{"x":{"dir":"input","width":8},"y":{"dir":"output","width":8},"z":{"dir":"output","width":8}},
          "body":[{"op":"output","args":{"y":"x"}}]
        }'''
        with pytest.raises(ValidationError, match="missing ports.*z"):
            _validate(raw)

    def test_extra_output_port(self):
        raw = '''{
          "name":"A",
          "ports":{"x":{"dir":"input","width":8},"y":{"dir":"output","width":8}},
          "body":[{"op":"output","args":{"y":"x","z":"x"}}]
        }'''
        with pytest.raises(ValidationError, match="extra ports.*z"):
            _validate(raw)

    def test_exact_coverage_passes(self):
        raw = '''{
          "name":"A",
          "ports":{"x":{"dir":"input","width":8},"y":{"dir":"output","width":8}},
          "body":[{"op":"output","args":{"y":"x"}}]
        }'''
        _validate(raw)  # should not raise

class TestInstance:
    def test_unknown_module(self):
        raw = '''{
          "name":"Top",
          "ports":{"x":{"dir":"input","width":8},"y":{"dir":"output","width":8}},
          "body":[
            {"id":["r"],"op":"instance","module":"NonExistent","args":{"a":"x"}},
            {"op":"output","args":{"y":"r"}}
          ]
        }'''
        with pytest.raises(ValidationError, match="unknown module"):
            _validate(raw)

    def test_instance_port_mismatch(self):
        raw = '''[
          {"name":"Sub","ports":{"i":{"dir":"input","width":8},"o":{"dir":"output","width":8}},
           "body":[{"op":"output","args":{"o":"i"}}]},
          {"name":"Top","ports":{"x":{"dir":"input","width":8},"y":{"dir":"output","width":8}},
           "body":[
             {"id":["r"],"op":"instance","module":"Sub","args":{"wrong":"x"}},
             {"op":"output","args":{"y":"r"}}
           ]}
        ]'''
        with pytest.raises(ValidationError, match="input ports mismatch"):
            _validate(raw)

    def test_instance_output_count_mismatch(self):
        raw = '''[
          {"name":"Sub","ports":{"i":{"dir":"input","width":8},"o":{"dir":"output","width":8}},
           "body":[{"op":"output","args":{"o":"i"}}]},
          {"name":"Top","ports":{"x":{"dir":"input","width":8},"y":{"dir":"output","width":8}},
           "body":[
             {"id":["r","s"],"op":"instance","module":"Sub","args":{"i":"x"}},
             {"op":"output","args":{"y":"r"}}
           ]}
        ]'''
        with pytest.raises(ValidationError, match="output id.*output port"):
            _validate(raw)

    def test_valid_instance(self):
        raw = '''[
          {"name":"Sub","ports":{"i":{"dir":"input","width":8},"o":{"dir":"output","width":8}},
           "body":[{"op":"output","args":{"o":"i"}}]},
          {"name":"Top","ports":{"x":{"dir":"input","width":8},"y":{"dir":"output","width":8}},
           "body":[
             {"id":["r"],"op":"instance","module":"Sub","args":{"i":"x"}},
             {"op":"output","args":{"y":"r"}}
           ]}
        ]'''
        _validate(raw)  # should not raise


class TestReg:
    def test_valid_basic_reg(self):
        raw = '''{
          "name":"A",
          "ports":{"clk":{"dir":"input","width":1},"d":{"dir":"input","width":8},"q":{"dir":"output","width":8}},
          "body":[
            {"id":"r","op":"reg","args":["d"],"clock":"clk"},
            {"op":"output","args":{"q":"r"}}
          ]
        }'''
        _validate(raw)  # should not raise

    def test_reg_undefined_data(self):
        raw = '''{
          "name":"A",
          "ports":{"clk":{"dir":"input","width":1},"q":{"dir":"output","width":8}},
          "body":[
            {"id":"r","op":"reg","args":["missing"],"clock":"clk"},
            {"op":"output","args":{"q":"r"}}
          ]
        }'''
        with pytest.raises(SSAError, match="undefined value 'missing'"):
            _validate(raw)

    def test_reg_undefined_clock(self):
        raw = '''{
          "name":"A",
          "ports":{"d":{"dir":"input","width":8},"q":{"dir":"output","width":8}},
          "body":[
            {"id":"r","op":"reg","args":["d"],"clock":"no_clk"},
            {"op":"output","args":{"q":"r"}}
          ]
        }'''
        with pytest.raises(SSAError, match="undefined value 'no_clk'"):
            _validate(raw)

    def test_reg_undefined_reset(self):
        raw = '''{
          "name":"A",
          "ports":{"clk":{"dir":"input","width":1},"d":{"dir":"input","width":8},"q":{"dir":"output","width":8}},
          "body":[
            {"id":"r","op":"reg","args":["d"],"clock":"clk","reset":"no_rst","resetValue":0},
            {"op":"output","args":{"q":"r"}}
          ]
        }'''
        with pytest.raises(SSAError, match="undefined value 'no_rst'"):
            _validate(raw)

    def test_reg_undefined_enable(self):
        raw = '''{
          "name":"A",
          "ports":{"clk":{"dir":"input","width":1},"d":{"dir":"input","width":8},"q":{"dir":"output","width":8}},
          "body":[
            {"id":"r","op":"reg","args":["d"],"clock":"clk","enable":"no_en"},
            {"op":"output","args":{"q":"r"}}
          ]
        }'''
        with pytest.raises(SSAError, match="undefined value 'no_en'"):
            _validate(raw)

    def test_reg_shadowing(self):
        raw = '''{
          "name":"A",
          "ports":{"clk":{"dir":"input","width":1},"d":{"dir":"input","width":8},"q":{"dir":"output","width":8}},
          "body":[
            {"id":"d","op":"reg","args":["d"],"clock":"clk"},
            {"op":"output","args":{"q":"d"}}
          ]
        }'''
        with pytest.raises(SSAError, match="already defined.*shadowing"):
            _validate(raw)


class TestCombinationalCycles:
    def test_self_loop(self):
        """a = add(x, a) — trivial self-loop."""
        raw = '''{
          "name":"A",
          "ports":{"x":{"dir":"input","width":8},"y":{"dir":"output","width":8}},
          "body":[
            {"id":"a","op":"add","args":["x","a"]},
            {"op":"output","args":{"y":"a"}}
          ]
        }'''
        with pytest.raises(CycleError, match="combinational cycle"):
            _validate(raw)

    def test_two_node_cycle(self):
        """a = add(x, b), b = add(x, a) — two-node cycle."""
        raw = '''{
          "name":"A",
          "ports":{"x":{"dir":"input","width":8},"y":{"dir":"output","width":8}},
          "body":[
            {"id":"a","op":"add","args":["x","b"]},
            {"id":"b","op":"add","args":["x","a"]},
            {"op":"output","args":{"y":"a"}}
          ]
        }'''
        with pytest.raises(CycleError, match="combinational cycle"):
            _validate(raw)

    def test_three_node_cycle(self):
        """a -> b -> c -> a cycle."""
        raw = '''{
          "name":"A",
          "ports":{"x":{"dir":"input","width":8},"y":{"dir":"output","width":8}},
          "body":[
            {"id":"a","op":"add","args":["x","c"]},
            {"id":"b","op":"add","args":["x","a"]},
            {"id":"c","op":"add","args":["x","b"]},
            {"op":"output","args":{"y":"a"}}
          ]
        }'''
        with pytest.raises(CycleError, match="combinational cycle"):
            _validate(raw)

    def test_cycle_through_mux(self):
        """a = add(x, m), m = mux(sel, a, x) — cycle through mux."""
        raw = '''{
          "name":"A",
          "ports":{"x":{"dir":"input","width":8},"sel":{"dir":"input","width":1},"y":{"dir":"output","width":8}},
          "body":[
            {"id":"a","op":"add","args":["x","m"]},
            {"id":"m","op":"mux","args":["sel","a","x"]},
            {"op":"output","args":{"y":"m"}}
          ]
        }'''
        with pytest.raises(CycleError, match="combinational cycle"):
            _validate(raw)

    def test_cycle_through_instance(self):
        """Cycle through an instance: a -> inst -> a."""
        raw = '''[
          {"name":"Sub","ports":{"i":{"dir":"input","width":8},"o":{"dir":"output","width":8}},
           "body":[{"op":"output","args":{"o":"i"}}]},
          {"name":"Top",
           "ports":{"x":{"dir":"input","width":8},"y":{"dir":"output","width":8}},
           "body":[
             {"id":"a","op":"add","args":["x","r"]},
             {"id":["r"],"op":"instance","module":"Sub","args":{"i":"a"}},
             {"op":"output","args":{"y":"r"}}
           ]}
        ]'''
        with pytest.raises(CycleError, match="combinational cycle"):
            _validate(raw)

    def test_mem_read_port_cycle(self):
        """Read output used to compute read address — combinational cycle."""
        raw = '''{
          "name":"A",
          "ports":{
            "clk":{"dir":"input","width":1},"rst":{"dir":"input","width":1},
            "x":{"dir":"input","width":4},"en":{"dir":"input","width":1},
            "y":{"dir":"output","width":8}
          },
          "body":[
            {"id":"addr","op":"add","args":["x","trunc"]},
            {"id":"trunc","op":"extract","args":["rd"],"lowBit":0,"width":4},
            {"id":["rd"],"op":"mem","width":8,"depth":16,
             "clock":"clk","reset":"rst",
             "reads":[{"addr":"addr","enable":"en"}],"writes":[]},
            {"op":"output","args":{"y":"rd"}}
          ]
        }'''
        with pytest.raises(CycleError, match="combinational cycle"):
            _validate(raw)

    def test_reg_breaks_cycle(self):
        """Valid counter pattern: acc = add(r, one), r = reg(acc)."""
        raw = '''{
          "name":"Counter",
          "ports":{"clk":{"dir":"input","width":1},"q":{"dir":"output","width":8}},
          "body":[
            {"id":"one","op":"constant","value":1,"width":8},
            {"id":"acc","op":"add","args":["r","one"]},
            {"id":"r","op":"reg","args":["acc"],"clock":"clk"},
            {"op":"output","args":{"q":"r"}}
          ]
        }'''
        _validate(raw)  # should not raise

    def test_pure_dag(self):
        """Simple pipeline DAG — no cycles."""
        raw = '''{
          "name":"A",
          "ports":{"x":{"dir":"input","width":8},"y":{"dir":"output","width":8}},
          "body":[
            {"id":"a","op":"add","args":["x","x"]},
            {"id":"b","op":"add","args":["a","a"]},
            {"op":"output","args":{"y":"b"}}
          ]
        }'''
        _validate(raw)  # should not raise
