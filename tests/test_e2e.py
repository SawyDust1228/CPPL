"""End-to-end tests: compile example JSON files and verify output."""

import os
import subprocess
import sys

import pytest

pytest.importorskip("pycde")

from cppl.codegen import generate_mlir, generate_verilog
from cppl.ir.parser import parse_design
from cppl.ir.validator import validate_design
from cppl.ir.infer import infer_widths

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..")
EXAMPLES_DIR = os.path.join(PROJECT_DIR, "examples")
CLI_CMD = ["circuitppl"]


def compile_example_file(path: str) -> str:
    with open(path) as f:
        raw = f.read()
    modules = parse_design(raw)
    validate_design(modules)
    widths = infer_widths(modules)
    return generate_mlir(modules, widths)


class TestExamples:
    def test_adder(self):
        mlir = compile_example_file(os.path.join(EXAMPLES_DIR, "adder.json"))
        assert "hw.module @Adder8" in mlir
        assert "comb.add" in mlir
        assert "hw.output" in mlir

    def test_alu(self):
        mlir = compile_example_file(os.path.join(EXAMPLES_DIR, "alu.json"))
        assert "hw.module @ALU" in mlir
        assert "comb.add" in mlir
        assert "comb.sub" in mlir
        assert "comb.and" in mlir
        assert "comb.xor" in mlir

    def test_hierarchy(self):
        mlir = compile_example_file(os.path.join(EXAMPLES_DIR, "hierarchy.json"))
        assert "hw.module @Adder8" in mlir
        assert "hw.module @Top" in mlir
        assert "hw.instance" in mlir
        assert "@Adder8" in mlir

    def test_register(self):
        mlir = compile_example_file(os.path.join(EXAMPLES_DIR, "register.json"))
        assert "hw.module @BasicReg" in mlir
        assert "hw.module @RegWithReset" in mlir
        assert "hw.module @RegWithEnable" in mlir
        assert "hw.module @RegFull" in mlir
        assert "seq.compreg" in mlir
        assert "seq.to_clock" in mlir


class TestCLI:
    def test_compile_to_stdout(self):
        input_path = os.path.join(EXAMPLES_DIR, "adder.json")
        result = subprocess.run(
            [*CLI_CMD, input_path, "--mlir"],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
        )
        assert result.returncode == 0
        assert "hw.module @Adder8" in result.stdout

    def test_validate_only(self):
        input_path = os.path.join(EXAMPLES_DIR, "adder.json")
        result = subprocess.run(
            [*CLI_CMD, input_path, "--validate-only"],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
        )
        assert result.returncode == 0
        assert "Validation passed" in result.stderr
        assert result.stdout == ""

    def test_stdin(self):
        input_path = os.path.join(EXAMPLES_DIR, "adder.json")
        with open(input_path) as f:
            input_data = f.read()
        result = subprocess.run(
            [*CLI_CMD, "-", "--mlir"],
            capture_output=True,
            text=True,
            input=input_data,
            cwd=PROJECT_DIR,
        )
        assert result.returncode == 0
        assert "hw.module @Adder8" in result.stdout

    def test_invalid_input(self):
        result = subprocess.run(
            [*CLI_CMD, "-"],
            capture_output=True,
            text=True,
            input="{bad json",
            cwd=PROJECT_DIR,
        )
        assert result.returncode == 1
        assert "Error" in result.stderr

    def test_output_file(self, tmp_path):
        input_path = os.path.join(EXAMPLES_DIR, "adder.json")
        output_path = str(tmp_path / "out.mlir")
        result = subprocess.run(
            [*CLI_CMD, input_path, "--mlir", "-o", output_path],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
        )
        assert result.returncode == 0
        with open(output_path) as f:
            content = f.read()
        assert "hw.module @Adder8" in content

    def test_dump_widths(self):
        input_path = os.path.join(EXAMPLES_DIR, "adder.json")
        result = subprocess.run(
            [*CLI_CMD, input_path, "--dump-widths", "--validate-only"],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
        )
        assert result.returncode == 0
        assert "i8" in result.stderr

    def test_verilog_adder(self):
        input_path = os.path.join(EXAMPLES_DIR, "adder.json")
        result = subprocess.run(
            [*CLI_CMD, input_path, "--verilog"],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
        )
        assert result.returncode == 0
        assert "module Adder8" in result.stdout
        assert "endmodule" in result.stdout

    def test_verilog_register(self):
        input_path = os.path.join(EXAMPLES_DIR, "register.json")
        result = subprocess.run(
            [*CLI_CMD, input_path, "--verilog"],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
        )
        assert result.returncode == 0
        assert "always_ff" in result.stdout or "always @" in result.stdout
        assert "endmodule" in result.stdout

    def test_verilog_output_file(self, tmp_path):
        input_path = os.path.join(EXAMPLES_DIR, "adder.json")
        output_path = str(tmp_path / "out.v")
        result = subprocess.run(
            [*CLI_CMD, input_path, "--verilog", "-o", output_path],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
        )
        assert result.returncode == 0
        with open(output_path) as f:
            content = f.read()
        assert "module Adder8" in content

    def test_top_mlir(self):
        input_path = os.path.join(EXAMPLES_DIR, "hierarchy.json")
        result = subprocess.run(
            [*CLI_CMD, input_path, "--mlir", "--top", "Top"],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
        )
        assert result.returncode == 0
        assert "hw.module @Top" in result.stdout
        assert "hw.module @Adder8" in result.stdout

    def test_top_verilog(self):
        input_path = os.path.join(EXAMPLES_DIR, "hierarchy.json")
        result = subprocess.run(
            [*CLI_CMD, input_path, "--verilog", "--top", "Top"],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
        )
        assert result.returncode == 0
        assert "module Top" in result.stdout
        assert "module Adder8" in result.stdout
        assert "endmodule" in result.stdout

    def test_top_unknown(self):
        input_path = os.path.join(EXAMPLES_DIR, "hierarchy.json")
        result = subprocess.run(
            [*CLI_CMD, input_path, "--mlir", "--top", "NoSuch"],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
        )
        assert result.returncode == 1
        assert "Error" in result.stderr


class TestRISCV:
    """End-to-end tests for the RISC-V design."""

    RISCV_PATH = os.path.join(EXAMPLES_DIR, "riscv.json")

    def test_validate(self):
        with open(self.RISCV_PATH) as f:
            raw = f.read()
        modules = parse_design(raw)
        validate_design(modules, strict=False)

    def test_mlir_all_modules(self):
        with open(self.RISCV_PATH) as f:
            raw = f.read()
        modules = parse_design(raw)
        validate_design(modules, strict=False)
        widths = infer_widths(modules)
        mlir = generate_mlir(modules, widths)
        # Check all expected modules are present
        for name in [
            "ALU",
            "BRU",
            "Immgen",
            "Control",
            "Regfile",
            "CSRGen",
            "Cache",
            "Datapath",
            "Core",
            "Tile",
        ]:
            assert f"hw.module @{name}" in mlir, f"Missing module @{name}"

    def test_mlir_top_tile(self):
        with open(self.RISCV_PATH) as f:
            raw = f.read()
        modules = parse_design(raw)
        validate_design(modules, strict=False)
        widths = infer_widths(modules)
        mlir = generate_mlir(modules, widths, top="Tile")
        assert "hw.module @Tile" in mlir
        assert "hw.module @Core" in mlir
        assert "hw.module @Cache" in mlir
        assert "hw.instance" in mlir

    def test_verilog_top_tile(self):
        with open(self.RISCV_PATH) as f:
            raw = f.read()
        modules = parse_design(raw)
        validate_design(modules, strict=False)
        widths = infer_widths(modules)
        verilog = generate_verilog(modules, widths, top="Tile")
        assert "module Tile" in verilog
        assert "module Core" in verilog
        assert "module Cache" in verilog
        assert "module Datapath" in verilog
        assert "module Control" in verilog
        assert "endmodule" in verilog

    def test_verilog_cli(self):
        with open(self.RISCV_PATH) as f:
            raw = f.read()
        modules = parse_design(raw)
        validate_design(modules, strict=False)
        widths = infer_widths(modules)
        verilog = generate_verilog(modules, widths, top="Tile")
        assert "module Tile" in verilog
        assert "endmodule" in verilog
