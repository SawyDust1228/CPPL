"""Tests for ir.codegen (and infer as a dependency)."""

import pytest

pytest.importorskip("pycde")

from cppl.codegen import generate_mlir, generate_verilog
from cppl.ir.errors import CodegenError, WidthError
from cppl.ir.parser import parse_design
from cppl.ir.validator import validate_design
from cppl.ir.infer import infer_widths


def compile_fixture(json_str: str) -> str:
    modules = parse_design(json_str)
    validate_design(modules)
    widths = infer_widths(modules)
    return generate_mlir(modules, widths)


def compile_verilog_fixture(json_str: str) -> str:
    modules = parse_design(json_str)
    validate_design(modules)
    widths = infer_widths(modules)
    return generate_verilog(modules, widths)


class TestWidthInference:
    def test_binary_width_mismatch(self):
        raw = """[
          {"name":"A","ports":{"x":{"dir":"input","width":8},"y":{"dir":"input","width":4},"o":{"dir":"output","width":8}},
           "body":[
             {"id":"r","op":"add","args":["x","y"]},
             {"op":"output","args":{"o":"r"}}
           ]}
        ]"""
        modules = parse_design(raw)
        validate_design(modules)
        with pytest.raises(WidthError, match="different widths"):
            infer_widths(modules)

    def test_extract_out_of_bounds(self):
        raw = """[
          {"name":"A","ports":{"x":{"dir":"input","width":8},"o":{"dir":"output","width":4}},
           "body":[
             {"id":"e","op":"extract","args":["x"],"lowBit":6,"width":4},
             {"op":"output","args":{"o":"e"}}
           ]}
        ]"""
        modules = parse_design(raw)
        validate_design(modules)
        with pytest.raises(WidthError, match="exceeds source width"):
            infer_widths(modules)

    def test_output_width_mismatch(self):
        raw = """[
          {"name":"A","ports":{"x":{"dir":"input","width":8},"o":{"dir":"output","width":4}},
           "body":[
             {"op":"output","args":{"o":"x"}}
           ]}
        ]"""
        modules = parse_design(raw)
        validate_design(modules)
        with pytest.raises(WidthError, match="output port.*expects width"):
            infer_widths(modules)

    def test_concat_width(self):
        raw = """[
          {"name":"A","ports":{"x":{"dir":"input","width":4},"y":{"dir":"input","width":4},"o":{"dir":"output","width":8}},
           "body":[
             {"id":"c","op":"concat","args":["x","y"]},
             {"op":"output","args":{"o":"c"}}
           ]}
        ]"""
        modules = parse_design(raw)
        validate_design(modules)
        widths = infer_widths(modules)
        assert widths["A"]["c"].width == 8


class TestCodegen:
    def test_adder(self):
        raw = """[
          {"name":"Adder8","ports":{"a":{"dir":"input","width":8},"b":{"dir":"input","width":8},"sum":{"dir":"output","width":8}},
           "body":[
             {"id":"result","op":"add","args":["a","b"]},
             {"op":"output","args":{"sum":"result"}}
           ]}
        ]"""
        mlir = compile_fixture(raw)
        assert "module {" in mlir or "builtin.module" in mlir
        assert "hw.module @Adder8" in mlir
        assert "comb.add" in mlir
        assert "hw.output" in mlir
        assert "i8" in mlir

    def test_constant(self):
        raw = """[
          {"name":"C","ports":{"o":{"dir":"output","width":8}},
           "body":[
             {"id":"c","op":"constant","value":42,"width":8},
             {"op":"output","args":{"o":"c"}}
           ]}
        ]"""
        mlir = compile_fixture(raw)
        assert "hw.constant 42 : i8" in mlir

    def test_constant_hex(self):
        raw = """[
          {"name":"C","ports":{"o":{"dir":"output","width":8}},
           "body":[
             {"id":"c","op":"constant","value":"0xff","width":8},
             {"op":"output","args":{"o":"c"}}
           ]}
        ]"""
        mlir = compile_fixture(raw)
        # CIRCT normalizes 0xff (255) to -1 for i8 (signed representation)
        assert "hw.constant" in mlir
        assert "i8" in mlir

    def test_not_lowering(self):
        raw = """[
          {"name":"N","ports":{"x":{"dir":"input","width":8},"o":{"dir":"output","width":8}},
           "body":[
             {"id":"n","op":"not","args":["x"]},
             {"op":"output","args":{"o":"n"}}
           ]}
        ]"""
        mlir = compile_fixture(raw)
        assert "hw.constant -1 : i8" in mlir
        assert "comb.xor" in mlir

    def test_neg_lowering(self):
        raw = """[
          {"name":"N","ports":{"x":{"dir":"input","width":8},"o":{"dir":"output","width":8}},
           "body":[
             {"id":"n","op":"neg","args":["x"]},
             {"op":"output","args":{"o":"n"}}
           ]}
        ]"""
        mlir = compile_fixture(raw)
        assert "hw.constant -1 : i8" in mlir
        assert "comb.xor" in mlir
        assert "hw.constant 1 : i8" in mlir
        assert "comb.add" in mlir

    def test_extract(self):
        raw = """[
          {"name":"E","ports":{"x":{"dir":"input","width":8},"o":{"dir":"output","width":4}},
           "body":[
             {"id":"e","op":"extract","args":["x"],"lowBit":2,"width":4},
             {"op":"output","args":{"o":"e"}}
           ]}
        ]"""
        mlir = compile_fixture(raw)
        assert "comb.extract" in mlir
        assert "from 2" in mlir
        assert "(i8) -> i4" in mlir

    def test_concat(self):
        raw = """[
          {"name":"C","ports":{"x":{"dir":"input","width":4},"y":{"dir":"input","width":4},"o":{"dir":"output","width":8}},
           "body":[
             {"id":"c","op":"concat","args":["x","y"]},
             {"op":"output","args":{"o":"c"}}
           ]}
        ]"""
        mlir = compile_fixture(raw)
        assert "comb.concat" in mlir
        assert "i4, i4" in mlir

    def test_all_binary_ops(self):
        ops = ["add", "sub", "mul", "div", "and", "or", "xor", "shl", "shr_u", "shr_s"]
        expected = [
            "comb.add",
            "comb.sub",
            "comb.mul",
            "comb.divu",
            "comb.and",
            "comb.or",
            "comb.xor",
            "comb.shl",
            "comb.shru",
            "comb.shrs",
        ]
        for op, circt_op in zip(ops, expected):
            raw = f"""[
              {{"name":"T","ports":{{"a":{{"dir":"input","width":8}},"b":{{"dir":"input","width":8}},"o":{{"dir":"output","width":8}}}},
               "body":[
                 {{"id":"r","op":"{op}","args":["a","b"]}},
                 {{"op":"output","args":{{"o":"r"}}}}
               ]}}
            ]"""
            mlir = compile_fixture(raw)
            assert circt_op in mlir, f"Expected {circt_op} for op '{op}'"

    def test_instance(self):
        raw = """[
          {"name":"Sub","ports":{"i":{"dir":"input","width":8},"o":{"dir":"output","width":8}},
           "body":[{"op":"output","args":{"o":"i"}}]},
          {"name":"Top","ports":{"x":{"dir":"input","width":8},"y":{"dir":"output","width":8}},
           "body":[
             {"id":["r"],"op":"instance","module":"Sub","args":{"i":"x"}},
             {"op":"output","args":{"y":"r"}}
           ]}
        ]"""
        mlir = compile_fixture(raw)
        assert "hw.instance" in mlir
        assert "@Sub" in mlir

    def test_reg_basic(self):
        raw = """[
          {"name":"R","ports":{"clk":{"dir":"input","width":1},"d":{"dir":"input","width":8},"q":{"dir":"output","width":8}},
           "body":[
             {"id":"r","op":"reg","args":["d"],"clock":"clk"},
             {"op":"output","args":{"q":"r"}}
           ]}
        ]"""
        mlir = compile_fixture(raw)
        assert "seq.compreg" in mlir
        assert "seq.to_clock" in mlir
        assert "i8" in mlir

    def test_reg_with_reset(self):
        raw = """[
          {"name":"R","ports":{"clk":{"dir":"input","width":1},"rst":{"dir":"input","width":1},"d":{"dir":"input","width":8},"q":{"dir":"output","width":8}},
           "body":[
             {"id":"r","op":"reg","args":["d"],"clock":"clk","reset":"rst","resetValue":0},
             {"op":"output","args":{"q":"r"}}
           ]}
        ]"""
        mlir = compile_fixture(raw)
        assert "seq.compreg" in mlir
        assert "seq.to_clock" in mlir

    def test_reg_with_enable(self):
        raw = """[
          {"name":"R","ports":{"clk":{"dir":"input","width":1},"en":{"dir":"input","width":1},"d":{"dir":"input","width":8},"q":{"dir":"output","width":8}},
           "body":[
             {"id":"r","op":"reg","args":["d"],"clock":"clk","enable":"en"},
             {"op":"output","args":{"q":"r"}}
           ]}
        ]"""
        mlir = compile_fixture(raw)
        assert "seq.compreg.ce" in mlir
        assert "seq.to_clock" in mlir

    def test_reg_full(self):
        raw = """[
          {"name":"R","ports":{"clk":{"dir":"input","width":1},"rst":{"dir":"input","width":1},"en":{"dir":"input","width":1},"d":{"dir":"input","width":8},"q":{"dir":"output","width":8}},
           "body":[
             {"id":"r","op":"reg","args":["d"],"clock":"clk","reset":"rst","resetValue":0,"enable":"en"},
             {"op":"output","args":{"q":"r"}}
           ]}
        ]"""
        mlir = compile_fixture(raw)
        assert "seq.compreg.ce" in mlir
        assert "seq.to_clock" in mlir


class TestRegWidthInference:
    def test_reg_clock_width_error(self):
        raw = """[
          {"name":"R","ports":{"clk":{"dir":"input","width":8},"d":{"dir":"input","width":8},"q":{"dir":"output","width":8}},
           "body":[
             {"id":"r","op":"reg","args":["d"],"clock":"clk"},
             {"op":"output","args":{"q":"r"}}
           ]}
        ]"""
        modules = parse_design(raw)
        validate_design(modules)
        with pytest.raises(WidthError, match="clock.*must be 1-bit"):
            infer_widths(modules)

    def test_reg_reset_width_error(self):
        raw = """[
          {"name":"R","ports":{"clk":{"dir":"input","width":1},"rst":{"dir":"input","width":8},"d":{"dir":"input","width":8},"q":{"dir":"output","width":8}},
           "body":[
             {"id":"r","op":"reg","args":["d"],"clock":"clk","reset":"rst","resetValue":0},
             {"op":"output","args":{"q":"r"}}
           ]}
        ]"""
        modules = parse_design(raw)
        validate_design(modules)
        with pytest.raises(WidthError, match="reset.*must be 1-bit"):
            infer_widths(modules)

    def test_reg_enable_width_error(self):
        raw = """[
          {"name":"R","ports":{"clk":{"dir":"input","width":1},"en":{"dir":"input","width":8},"d":{"dir":"input","width":8},"q":{"dir":"output","width":8}},
           "body":[
             {"id":"r","op":"reg","args":["d"],"clock":"clk","enable":"en"},
             {"op":"output","args":{"q":"r"}}
           ]}
        ]"""
        modules = parse_design(raw)
        validate_design(modules)
        with pytest.raises(WidthError, match="enable.*must be 1-bit"):
            infer_widths(modules)

    def test_reg_output_width_equals_data_width(self):
        raw = """[
          {"name":"R","ports":{"clk":{"dir":"input","width":1},"d":{"dir":"input","width":16},"q":{"dir":"output","width":16}},
           "body":[
             {"id":"r","op":"reg","args":["d"],"clock":"clk"},
             {"op":"output","args":{"q":"r"}}
           ]}
        ]"""
        modules = parse_design(raw)
        validate_design(modules)
        widths = infer_widths(modules)
        assert widths["R"]["r"].width == 16


class TestVerilog:
    def test_combinational_verilog(self):
        raw = """[
          {"name":"Adder8","ports":{"a":{"dir":"input","width":8},"b":{"dir":"input","width":8},"sum":{"dir":"output","width":8}},
           "body":[
             {"id":"result","op":"add","args":["a","b"]},
             {"op":"output","args":{"sum":"result"}}
           ]}
        ]"""
        verilog = compile_verilog_fixture(raw)
        assert "module Adder8" in verilog
        assert "input" in verilog
        assert "output" in verilog
        assert "endmodule" in verilog

    def test_register_verilog(self):
        raw = """[
          {"name":"RegMod","ports":{"clk":{"dir":"input","width":1},"rst":{"dir":"input","width":1},"d":{"dir":"input","width":8},"q":{"dir":"output","width":8}},
           "body":[
             {"id":"r","op":"reg","args":["d"],"clock":"clk","reset":"rst","resetValue":0},
             {"op":"output","args":{"q":"r"}}
           ]}
        ]"""
        verilog = compile_verilog_fixture(raw)
        assert "module RegMod" in verilog
        assert "always_ff" in verilog or "always @" in verilog
        assert "endmodule" in verilog


HIERARCHY_JSON = """[
  {"name":"Leaf","ports":{"i":{"dir":"input","width":8},"o":{"dir":"output","width":8}},
   "body":[{"op":"output","args":{"o":"i"}}]},
  {"name":"Mid","ports":{"x":{"dir":"input","width":8},"y":{"dir":"output","width":8}},
   "body":[
     {"id":["r"],"op":"instance","module":"Leaf","args":{"i":"x"}},
     {"op":"output","args":{"y":"r"}}
   ]},
  {"name":"Top","ports":{"a":{"dir":"input","width":8},"b":{"dir":"output","width":8}},
   "body":[
     {"id":["s"],"op":"instance","module":"Mid","args":{"x":"a"}},
     {"op":"output","args":{"b":"s"}}
   ]},
  {"name":"Unrelated","ports":{"p":{"dir":"input","width":4},"q":{"dir":"output","width":4}},
   "body":[{"op":"output","args":{"q":"p"}}]}
]"""


class TestTop:
    def test_top_filters_mlir(self):
        modules = parse_design(HIERARCHY_JSON)
        validate_design(modules)
        widths = infer_widths(modules)
        mlir = generate_mlir(modules, widths, top="Top")
        assert "hw.module @Top" in mlir
        assert "hw.module @Mid" in mlir
        assert "hw.module @Leaf" in mlir
        assert "Unrelated" not in mlir

    def test_top_filters_verilog(self):
        modules = parse_design(HIERARCHY_JSON)
        validate_design(modules)
        widths = infer_widths(modules)
        verilog = generate_verilog(modules, widths, top="Top")
        assert "module Top" in verilog
        assert "module Mid" in verilog
        assert "module Leaf" in verilog
        assert "Unrelated" not in verilog

    def test_top_single_module(self):
        modules = parse_design(HIERARCHY_JSON)
        validate_design(modules)
        widths = infer_widths(modules)
        mlir = generate_mlir(modules, widths, top="Unrelated")
        assert "hw.module @Unrelated" in mlir
        assert "Top" not in mlir
        assert "Mid" not in mlir
        assert "Leaf" not in mlir

    def test_top_unknown_module(self):
        modules = parse_design(HIERARCHY_JSON)
        validate_design(modules)
        widths = infer_widths(modules)
        with pytest.raises(CodegenError, match="not found"):
            generate_mlir(modules, widths, top="NoSuchModule")


class TestNewOps:
    def test_mux(self):
        raw = """[
          {"name":"M","ports":{"s":{"dir":"input","width":1},"a":{"dir":"input","width":8},"b":{"dir":"input","width":8},"o":{"dir":"output","width":8}},
           "body":[
             {"id":"r","op":"mux","args":["s","a","b"]},
             {"op":"output","args":{"o":"r"}}
           ]}
        ]"""
        mlir = compile_fixture(raw)
        assert "comb.mux" in mlir
        assert "i8" in mlir

    def test_mux_width_inference(self):
        raw = """[
          {"name":"M","ports":{"s":{"dir":"input","width":1},"a":{"dir":"input","width":8},"b":{"dir":"input","width":8},"o":{"dir":"output","width":8}},
           "body":[
             {"id":"r","op":"mux","args":["s","a","b"]},
             {"op":"output","args":{"o":"r"}}
           ]}
        ]"""
        modules = parse_design(raw)
        validate_design(modules)
        widths = infer_widths(modules)
        assert widths["M"]["r"].width == 8

    def test_mux_sel_must_be_1bit(self):
        raw = """[
          {"name":"M","ports":{"s":{"dir":"input","width":2},"a":{"dir":"input","width":8},"b":{"dir":"input","width":8},"o":{"dir":"output","width":8}},
           "body":[
             {"id":"r","op":"mux","args":["s","a","b"]},
             {"op":"output","args":{"o":"r"}}
           ]}
        ]"""
        modules = parse_design(raw)
        validate_design(modules)
        with pytest.raises(WidthError, match="mux selector.*must be 1-bit"):
            infer_widths(modules)

    def test_mux_operand_width_mismatch(self):
        raw = """[
          {"name":"M","ports":{"s":{"dir":"input","width":1},"a":{"dir":"input","width":8},"b":{"dir":"input","width":4},"o":{"dir":"output","width":8}},
           "body":[
             {"id":"r","op":"mux","args":["s","a","b"]},
             {"op":"output","args":{"o":"r"}}
           ]}
        ]"""
        modules = parse_design(raw)
        validate_design(modules)
        with pytest.raises(WidthError, match="mux true/false.*different widths"):
            infer_widths(modules)

    def test_compare_ops(self):
        ops = ["eq", "ne", "lt_s", "lt_u", "ge_s", "ge_u"]
        for op in ops:
            raw = f"""[
              {{"name":"T","ports":{{"a":{{"dir":"input","width":8}},"b":{{"dir":"input","width":8}},"o":{{"dir":"output","width":1}}}},
               "body":[
                 {{"id":"r","op":"{op}","args":["a","b"]}},
                 {{"op":"output","args":{{"o":"r"}}}}
               ]}}
            ]"""
            mlir = compile_fixture(raw)
            assert "comb.icmp" in mlir, f"Expected comb.icmp for op '{op}'"

    def test_compare_output_width(self):
        raw = """[
          {"name":"T","ports":{"a":{"dir":"input","width":32},"b":{"dir":"input","width":32},"o":{"dir":"output","width":1}},
           "body":[
             {"id":"r","op":"eq","args":["a","b"]},
             {"op":"output","args":{"o":"r"}}
           ]}
        ]"""
        modules = parse_design(raw)
        validate_design(modules)
        widths = infer_widths(modules)
        assert widths["T"]["r"].width == 1

    def test_or_reduce(self):
        raw = """[
          {"name":"T","ports":{"a":{"dir":"input","width":8},"o":{"dir":"output","width":1}},
           "body":[
             {"id":"r","op":"or_reduce","args":["a"]},
             {"op":"output","args":{"o":"r"}}
           ]}
        ]"""
        mlir = compile_fixture(raw)
        assert "comb.icmp" in mlir

    def test_and_reduce(self):
        raw = """[
          {"name":"T","ports":{"a":{"dir":"input","width":8},"o":{"dir":"output","width":1}},
           "body":[
             {"id":"r","op":"and_reduce","args":["a"]},
             {"op":"output","args":{"o":"r"}}
           ]}
        ]"""
        mlir = compile_fixture(raw)
        assert "comb.icmp" in mlir

    def test_reduce_output_width(self):
        raw = """[
          {"name":"T","ports":{"a":{"dir":"input","width":16},"o":{"dir":"output","width":1}},
           "body":[
             {"id":"r","op":"or_reduce","args":["a"]},
             {"op":"output","args":{"o":"r"}}
           ]}
        ]"""
        modules = parse_design(raw)
        validate_design(modules)
        widths = infer_widths(modules)
        assert widths["T"]["r"].width == 1

    def test_sext(self):
        raw = """[
          {"name":"T","ports":{"a":{"dir":"input","width":8},"o":{"dir":"output","width":32}},
           "body":[
             {"id":"r","op":"sext","args":["a"],"width":32},
             {"op":"output","args":{"o":"r"}}
           ]}
        ]"""
        mlir = compile_fixture(raw)
        assert "comb.replicate" in mlir
        assert "comb.concat" in mlir

    def test_zext(self):
        raw = """[
          {"name":"T","ports":{"a":{"dir":"input","width":8},"o":{"dir":"output","width":32}},
           "body":[
             {"id":"r","op":"zext","args":["a"],"width":32},
             {"op":"output","args":{"o":"r"}}
           ]}
        ]"""
        mlir = compile_fixture(raw)
        assert "hw.constant 0 : i24" in mlir
        assert "comb.concat" in mlir

    def test_cast_width_inference(self):
        raw = """[
          {"name":"T","ports":{"a":{"dir":"input","width":8},"o":{"dir":"output","width":32}},
           "body":[
             {"id":"r","op":"sext","args":["a"],"width":32},
             {"op":"output","args":{"o":"r"}}
           ]}
        ]"""
        modules = parse_design(raw)
        validate_design(modules)
        widths = infer_widths(modules)
        assert widths["T"]["r"].width == 32

    def test_cast_target_too_small(self):
        raw = """[
          {"name":"T","ports":{"a":{"dir":"input","width":8},"o":{"dir":"output","width":4}},
           "body":[
             {"id":"r","op":"sext","args":["a"],"width":4},
             {"op":"output","args":{"o":"r"}}
           ]}
        ]"""
        modules = parse_design(raw)
        validate_design(modules)
        with pytest.raises(WidthError, match="target width.*less than"):
            infer_widths(modules)

    def test_sext_same_width(self):
        raw = """[
          {"name":"T","ports":{"a":{"dir":"input","width":8},"o":{"dir":"output","width":8}},
           "body":[
             {"id":"r","op":"sext","args":["a"],"width":8},
             {"op":"output","args":{"o":"r"}}
           ]}
        ]"""
        mlir = compile_fixture(raw)
        assert "hw.module @T" in mlir

    def test_divs(self):
        raw = """[
          {"name":"T","ports":{"a":{"dir":"input","width":8},"b":{"dir":"input","width":8},"o":{"dir":"output","width":8}},
           "body":[
             {"id":"r","op":"div_s","args":["a","b"]},
             {"op":"output","args":{"o":"r"}}
           ]}
        ]"""
        mlir = compile_fixture(raw)
        assert "comb.divs" in mlir

    def test_modu(self):
        raw = """[
          {"name":"T","ports":{"a":{"dir":"input","width":8},"b":{"dir":"input","width":8},"o":{"dir":"output","width":8}},
           "body":[
             {"id":"r","op":"mod_u","args":["a","b"]},
             {"op":"output","args":{"o":"r"}}
           ]}
        ]"""
        mlir = compile_fixture(raw)
        assert "comb.modu" in mlir

    def test_mods(self):
        raw = """[
          {"name":"T","ports":{"a":{"dir":"input","width":8},"b":{"dir":"input","width":8},"o":{"dir":"output","width":8}},
           "body":[
             {"id":"r","op":"mod_s","args":["a","b"]},
             {"op":"output","args":{"o":"r"}}
           ]}
        ]"""
        mlir = compile_fixture(raw)
        assert "comb.mods" in mlir

    def test_gt_le_compare_ops(self):
        ops = ["gt_s", "gt_u", "le_s", "le_u"]
        for op in ops:
            raw = f"""[
              {{"name":"T","ports":{{"a":{{"dir":"input","width":8}},"b":{{"dir":"input","width":8}},"o":{{"dir":"output","width":1}}}},
               "body":[
                 {{"id":"r","op":"{op}","args":["a","b"]}},
                 {{"op":"output","args":{{"o":"r"}}}}
               ]}}
            ]"""
            mlir = compile_fixture(raw)
            assert "comb.icmp" in mlir, f"Expected comb.icmp for op '{op}'"

    def test_xor_reduce(self):
        raw = """[
          {"name":"T","ports":{"a":{"dir":"input","width":8},"o":{"dir":"output","width":1}},
           "body":[
             {"id":"r","op":"xor_reduce","args":["a"]},
             {"op":"output","args":{"o":"r"}}
           ]}
        ]"""
        mlir = compile_fixture(raw)
        assert "comb.parity" in mlir

    def test_reverse(self):
        raw = """[
          {"name":"T","ports":{"a":{"dir":"input","width":8},"o":{"dir":"output","width":8}},
           "body":[
             {"id":"r","op":"reverse","args":["a"]},
             {"op":"output","args":{"o":"r"}}
           ]}
        ]"""
        mlir = compile_fixture(raw)
        assert "comb.reverse" in mlir


MEM_BASIC_JSON = """[
  {"name":"MemTest",
   "ports":{
     "clk":{"dir":"input","width":1},"rst":{"dir":"input","width":1},
     "addr":{"dir":"input","width":10},"wdata":{"dir":"input","width":32},
     "wen":{"dir":"input","width":1},"ren":{"dir":"input","width":1},
     "rdata":{"dir":"output","width":32}
   },
   "body":[
     {"id":["rd"],"op":"mem","width":32,"depth":1024,
      "clock":"clk","reset":"rst",
      "reads":[{"addr":"addr","enable":"ren"}],
      "writes":[{"addr":"addr","data":"wdata","enable":"wen"}]},
     {"op":"output","args":{"rdata":"rd"}}
   ]}
]"""


class TestMem:
    def test_mem_basic(self):
        mlir = compile_fixture(MEM_BASIC_JSON)
        assert "seq.hlmem" in mlir
        assert "seq.read" in mlir
        assert "seq.write" in mlir

    def test_mem_verilog(self):
        verilog = compile_verilog_fixture(MEM_BASIC_JSON)
        assert "module MemTest" in verilog
        assert "reg" in verilog.lower()
        assert "always_ff" in verilog or "always @" in verilog
        assert "endmodule" in verilog

    def test_mem_multi_read(self):
        raw = """[
          {"name":"M",
           "ports":{
             "clk":{"dir":"input","width":1},"rst":{"dir":"input","width":1},
             "addr1":{"dir":"input","width":5},"addr2":{"dir":"input","width":5},
             "ren1":{"dir":"input","width":1},"ren2":{"dir":"input","width":1},
             "waddr":{"dir":"input","width":5},"wdata":{"dir":"input","width":32},
             "wen":{"dir":"input","width":1},
             "rd1":{"dir":"output","width":32},"rd2":{"dir":"output","width":32}
           },
           "body":[
             {"id":["r1","r2"],"op":"mem","width":32,"depth":32,
              "clock":"clk","reset":"rst",
              "reads":[
                {"addr":"addr1","enable":"ren1"},
                {"addr":"addr2","enable":"ren2"}
              ],
              "writes":[{"addr":"waddr","data":"wdata","enable":"wen"}]},
             {"op":"output","args":{"rd1":"r1","rd2":"r2"}}
           ]}
        ]"""
        mlir = compile_fixture(raw)
        assert "seq.hlmem" in mlir
        # Two read ports
        assert mlir.count("seq.read") == 2

    def test_mem_width_inference(self):
        raw = """[
          {"name":"M",
           "ports":{
             "clk":{"dir":"input","width":1},"rst":{"dir":"input","width":1},
             "addr":{"dir":"input","width":5},"wdata":{"dir":"input","width":16},
             "wen":{"dir":"input","width":1},"ren":{"dir":"input","width":1},
             "rdata":{"dir":"output","width":16}
           },
           "body":[
             {"id":["rd"],"op":"mem","width":16,"depth":32,
              "clock":"clk","reset":"rst",
              "reads":[{"addr":"addr","enable":"ren"}],
              "writes":[{"addr":"addr","data":"wdata","enable":"wen"}]},
             {"op":"output","args":{"rdata":"rd"}}
           ]}
        ]"""
        modules = parse_design(raw)
        validate_design(modules)
        widths = infer_widths(modules)
        assert widths["M"]["rd"].width == 16
