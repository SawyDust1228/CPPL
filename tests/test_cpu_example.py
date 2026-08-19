"""Synchronization checks for the RV32I CPU example and problem image."""

from __future__ import annotations

import importlib.util
from pathlib import Path


CPU_EXAMPLE = Path(__file__).parents[1] / "examples" / "python" / "cpu.py"


def load_cpu_example():
    spec = importlib.util.spec_from_file_location("cppl_cpu_example", CPU_EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_pattern_loads_current_problem_image():
    cpu = load_cpu_example()
    expected_words = {
        index: int(word, 16)
        for index, word in enumerate(
            cpu.INSTRUCTION_IMAGE.read_text(encoding="utf-8").split()
        )
    }

    pattern = cpu.CPU.patterns[0]
    assert dict(pattern.fixtures[0].words) == expected_words
    assert dict(pattern.fixtures[0].words) == cpu.PROGRAM_WORDS


def test_cpu_pattern_checks_current_program_final_state():
    cpu = load_cpu_example()
    pattern = cpu.CPU.patterns[0]
    final_step = pattern.steps[-1]

    assert len(cpu.PROGRAM_WORDS) == 46
    assert final_step.inputs == {"clk": 1}
    assert final_step.probes == {
        "pc0.pc": 0x0B4,
        "reg_file0.reg_file[0]": 0,
        "reg_file0.reg_file[20]": 2,
        "reg_file0.reg_file[21]": 3,
        "reg_file0.reg_file[22]": 4,
        "mem0.mem[66]": 2,
    }
