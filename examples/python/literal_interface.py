from cppl import module, In, Out, Design


@module
def Inc8(x: In[8]) -> {"y": Out[8]}:
    """y equals x plus 1."""
    pass


@module
def LiteralInterfaceTop(a: In[8], b: In[8]) -> {"out": Out[8]}:
    inc = Inc8("sum_ab")
    return f"""
    Define sum_ab as a plus b.
    Instantiate Inc8 using sum_ab as its input.
    out is {inc}.
    """


design = Design()
design.add(LiteralInterfaceTop)

if __name__ == "__main__":
    print(design.to_verilog())
