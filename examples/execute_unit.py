from cppl import module, In, Out, Design


@module
def Add32(a: In[32], b: In[32]) -> {"sum": Out[32]}:
    """sum equals a plus b."""
    pass


@module
def LogicUnit32(op: In[2], a: In[32], b: In[32]) -> {"out": Out[32]}:
    """Select one 32-bit logic result from op.

    - 00: out = a & b
    - 01: out = a | b
    - 10: out = a ^ b
    - 11: out = bitwise not of a
    """
    pass


@module
def ALU32(op: In[2], a: In[32], b: In[32]) -> {"result": Out[32], "zero": Out[1]}:
    logic = LogicUnit32(op, a, b)
    add = Add32(a, b)
    return f"""
    Build a 32-bit ALU using the pre-instantiated helper modules.
    Use {logic} for op values 00, 01, and 10.
    Use {add} when op is 11.
    result is the selected value.
    zero is 1 when result equals 0, otherwise 0.
    """


@module
def BranchUnit32(kind: In[2], lhs: In[32], rhs: In[32]) -> {"take": Out[1]}:
    """Generate a branch decision.

    - 00: take = 0
    - 01: take = 1 when lhs equals rhs
    - 10: take = 1 when lhs does not equal rhs
    - 11: take = 1 when lhs is unsigned less than rhs
    """
    pass


@module
def ExecuteUnit32(
    pc: In[32],
    imm: In[32],
    src_a: In[32],
    src_b: In[32],
    alu_op: In[2],
    branch_kind: In[2],
) -> {"alu_result": Out[32], "next_pc": Out[32], "branch_taken": Out[1]}:
    alu = ALU32(alu_op, src_a, src_b)
    branch_taken = BranchUnit32(branch_kind, src_a, src_b)
    target = Add32(pc, imm)
    return f"""
    Compose an execute-stage datapath from the submodules.
    alu_result is {alu.result}.
    branch_taken is {branch_taken}.
    If {branch_taken} is 1, next_pc is {target}; otherwise next_pc is pc plus 4.
    """


design = Design()
design.add(ExecuteUnit32)

if __name__ == "__main__":
    print(design.to_verilog())
