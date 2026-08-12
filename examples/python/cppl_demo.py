"""CPPL demo: define hardware modules as decorated Python functions."""

from cppl import Case, Design, In, Out, module


@module(
    patterns=[
        Case(
            name="basic_add",
            inputs={"a": 1, "b": 2},
            outputs={"sum": 3},
        ),
        Case(
            name="wraparound",
            inputs={"a": 255, "b": 1},
            outputs={"sum": 0},
        ),
    ]
)
def Adder8(a: In[8], b: In[8]) -> {"sum": Out[8]}:
    """sum equals a plus b (8-bit addition)."""
    pass


@module(
    patterns=[
        Case(
            name="addition",
            inputs={"op_code": 0, "op_a": 5, "op_b": 3},
            outputs={"res": 8, "zero": 0},
        ),
        Case(
            name="subtraction_to_zero",
            inputs={"op_code": 1, "op_a": 9, "op_b": 9},
            outputs={"res": 0, "zero": 1},
        ),
        Case(
            name="bitwise_and",
            inputs={"op_code": 2, "op_a": 0xF0, "op_b": 0x3C},
            outputs={"res": 0x30, "zero": 0},
        ),
        Case(
            name="bitwise_or",
            inputs={"op_code": 3, "op_a": 0x80, "op_b": 0x01},
            outputs={"res": 0x81, "zero": 0},
        ),
    ]
)
def ALU(op_code: In[2], op_a: In[8], op_b: In[8]) -> {"res": Out[8], "zero": Out[1]}:
    return f"""
    Simple ALU that uses an Adder8 instance for addition.
    Based on op_code (2-bit selector):
    - 00: res = {Adder8(op_a, op_b)} (result from Adder8 instance)
    - 01: res = op_a - op_b
    - 10: res = op_a & op_b (bitwise AND)
    - 11: res = op_a | op_b (bitwise OR)
    zero is 1 when res equals 0, otherwise 0.
    """


design = Design()
design.add(ALU)

if __name__ == "__main__":
    print(design.to_verilog())
