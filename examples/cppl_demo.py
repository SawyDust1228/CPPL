"""CPPL demo: define hardware modules as decorated Python functions."""

from cppl import module, In, Out, Design


@module
def Adder8(a: In[8], b: In[8]) -> {"sum": Out[8]}:
    """out equals a plus b (8-bit addition)."""
    pass


@module
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
