# CircuitPPL

A Python DSL for hardware description with LLM-powered compilation. Define hardware modules as Python functions, let an LLM generate the logic, and compile to MLIR or Verilog.

## Architecture

```
Python DSL (@module)  →  LLM Compilation  →  JSON-IR  →  MLIR / Verilog
```

- **Frontend**: `@module` decorator extracts ports and logic descriptions from Python functions
- **Compiler**: LLM generates JSON-IR body from natural-language descriptions (with validation feedback loop)
- **IR**: SSA-based intermediate representation with parsing, validation, and bit-width inference
- **Codegen**: MLIR/CIRCT emission via pycde, with Verilog export

## Quick Start

### Define Modules

```python
from cppl import module, In, Out, Design

@module
def Adder8(a: In[8], b: In[8]) -> Out[8]:
    """out equals a plus b (8-bit addition)."""
    pass

@module
def ALU(op_code: In[2], op_a: In[8], op_b: In[8]) -> {"res": Out[8], "zero": Out[1]}:
    adder8_out = Adder8(op_a, op_b)
    return f"""
    Based on op_code:
    - 00: res = {adder8_out} (from Adder8 instance)
    - 01: res = op_a - op_b
    - 10: res = op_a & op_b
    - 11: res = op_a | op_b
    zero is 1 when res equals 0, otherwise 0.
    """

design = Design()
design.add(ALU)  # automatically discovers and adds Adder8
print(design.to_verilog())
```

Key points:
- `In[N]` / `Out[N]` specify N-bit input/output ports
- Single output uses `-> Out[N]` (named `"out"`), multiple outputs use `-> {"name": Out[N], ...}`
- Docstrings describe the logic in natural language for the LLM
- Modules can instantiate other modules by calling them; use f-string returns to reference instance outputs
- `Design.add()` recursively discovers and adds all sub-modules — only the top-level module needs to be added

### Run the Demo

**Install (editable / development mode):**

```bash
pip install -e .
```

Run the example as a normal Python script:

```bash
python examples/cppl_demo.py
```

## JSON-IR Format

The intermediate representation is a JSON array of modules. Each module has `name`, `ports`, and a `body` of SSA operations:

```json
[{
  "name": "Adder8",
  "ports": {
    "a": {"dir": "input", "width": 8},
    "b": {"dir": "input", "width": 8},
    "sum": {"dir": "output", "width": 8}
  },
  "body": [
    {"id": "result", "op": "add", "args": ["a", "b"]},
    {"op": "output", "args": {"sum": "result"}}
  ]
}]
```

Supported operations: `constant`, `add`, `sub`, `mul`, `div`, `div_s`, `mod_u`, `mod_s`, `and`, `or`, `xor`, `not`, `neg`, `reverse`, `shl`, `shr_u`, `shr_s`, `eq`, `ne`, `lt_s`, `lt_u`, `ge_s`, `ge_u`, `gt_s`, `gt_u`, `le_s`, `le_u`, `or_reduce`, `and_reduce`, `xor_reduce`, `concat`, `extract`, `mux`, `sext`, `zext`, `reg`, `mem`, `instance`, `output`.

### Memory (`mem`)

The `mem` op creates register-array memories with configurable read/write ports:

```json
{
  "id": ["rdata"],
  "op": "mem",
  "width": 32,
  "depth": 1024,
  "clock": "clk",
  "reset": "rst",
  "reads": [{"addr": "raddr", "enable": "ren"}],
  "writes": [{"addr": "waddr", "data": "wdata", "enable": "wen"}]
}
```

- `id` has one entry per read port; each output is `width` bits wide
- `reads`: combinational read ports (latency 0); `writes`: synchronous write ports (latency 1)
- Generates CIRCT `seq.hlmem` ops, lowered to `sv.reg` arrays in Verilog

## Dependencies

- [appl](https://github.com/appl-team/appl) — LLM prompt framework
- [pycde](https://github.com/llvm/circt) — Python CIRCT bindings for MLIR/Verilog emission
- An LLM API key configured for your chosen provider

### Environment Configuration

Create a root `.env` file for your provider credentials and model selection.
CircuitPPL reads LLM configuration from `.env`; `appl.yaml` is not required.

Recommended generic variables:

```dotenv
LLM_PROVIDER=openai
LLM_MODEL=openai/gpt-4.1
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://api.openai.com/v1
```

You can also use provider-specific variables when you prefer:

```dotenv
LLM_PROVIDER=deepseek
DEEPSEEK_MODEL=deepseek/deepseek-chat
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

Notes:

- `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, and `LLM_BASE_URL` are the primary, provider-agnostic settings.
- If `LLM_PROVIDER` is set, CircuitPPL also checks `${PROVIDER}_MODEL`, `${PROVIDER}_API_KEY`, and `${PROVIDER}_BASE_URL`.
- Existing fallbacks like `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_BASE_URL`, and `OPENAI_BASE_URL` still work.
- `LLM_MODEL` is required for DSL compilation because there is no `appl.yaml` fallback.
- Optional generation overrides are available via `LLM_TEMPERATURE`, `LLM_TOP_P`, and `LLM_REASONING_EFFORT`. By default CircuitPPL does not force `temperature`, which avoids compatibility issues with models such as `gpt-5`.

## Project Structure

```
cppl/
├── frontend/          # DSL types, @module decorator, LLM compiler, prompts
├── ir/                # JSON-IR models, parser, validator, width inference
├── codegen/           # MLIR/Verilog generation via CIRCT
examples/              # Python DSL examples
pyproject.toml         # Package configuration
```
