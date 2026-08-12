from cppl import module, In, Out, Clock, Design


@module
def reg_file(
    clk: Clock,
    rst: In[1],
    A1: In[5],
    A2: In[5],
    A3: In[5],
    WD: In[32],
    WE: In[1],
) -> {"RD1": Out[32], "RD2": Out[32]}:
    """Implement a 32-entry, 32-bit RISC-V register file.

    Use one internal memory named reg_file with width 32 and depth 32.
    It has two combinational read ports addressed by A1 and A2.
    It has one synchronous write port addressed by A3.
    Register x0 must always read as zero, and writes to x0 must be ignored.
    Reset initializes the storage to zero.
    """
    pass


@module
def pc(clk: Clock, rst: In[1], JUMP: In[1], JUMP_PC: In[32]) -> {"pc": Out[32]}:
    """Implement the program counter register.

    On reset, pc becomes 0.
    Otherwise, on each rising edge of clk:
    - if JUMP is 1, pc becomes JUMP_PC
    - otherwise, pc becomes pc + 4
    """
    pass


@module
def imm(inst: In[32]) -> {"out": Out[32]}:
    """Generate the RV32I immediate value for inst.

    Decode opcode inst[6:0].
    I-type immediates are sign-extended from inst[31:20].
    S-type immediates are sign-extended from {inst[31:25], inst[11:7]}.
    B-type immediates are sign-extended from {inst[31], inst[7], inst[30:25], inst[11:8], 1'b0}.
    U-type immediates are {inst[31:12], 12'b0}.
    J-type immediates are sign-extended from {inst[31], inst[19:12], inst[20], inst[30:21], 1'b0}.
    Use U-type for LUI and AUIPC, B-type for branches, J-type for JAL,
    I-type for JALR, loads, and arithmetic I-type instructions, and S-type for stores.
    Output zero for unsupported opcodes.
    """
    pass


@module
def branch(REG1: In[32], REG2: In[32], Type: In[3]) -> {"BrE": Out[1]}:
    """Compute a branch decision.

    Type encoding:
    - 000: not taken
    - 001: BEQ, REG1 == REG2
    - 010: BNE, REG1 != REG2
    - 011: BLT, signed REG1 < signed REG2
    - 100: BGE, signed REG1 >= signed REG2
    - 101: BLTU, unsigned REG1 < unsigned REG2
    - 110: BGEU, unsigned REG1 >= unsigned REG2
    Other values produce 0.
    """
    pass


@module
def alu(SrcA: In[32], SrcB: In[32], func: In[4]) -> {"ALUout": Out[32]}:
    """Implement the RV32I ALU.

    func encoding:
    - 0000: add
    - 1000: sub
    - 0001: shift left logical by SrcB[4:0]
    - 0101: shift right logical by SrcB[4:0]
    - 1101: shift right arithmetic by SrcB[4:0]
    - 0010: signed less-than, output 32'd1 or 32'd0
    - 0011: unsigned less-than, output 32'd1 or 32'd0
    - 0100: xor
    - 0110: or
    - 0111: and
    - 1110: pass SrcB
    Default output is zero.
    """
    pass


@module
def mem(
    clk: Clock,
    im_addr: In[32],
    dm_rd_ctrl: In[3],
    dm_wr_ctrl: In[2],
    dm_addr: In[32],
    dm_din: In[32],
) -> {"im_dout": Out[32], "dm_dout": Out[32]}:
    """Describe a unified instruction and data storage block.

    The block contains one internal 32-bit-wide storage array named mem with
    4096 word entries. Initialize that storage from ./problem/inst.dat using
    hexadecimal text. The instruction side reads the word indexed by
    im_addr[13:2] and drives im_dout. The data side reads the word indexed by
    dm_addr[13:2] in the same cycle. Data writes are synchronous on clk.

    dm_rd_ctrl encoding:
    - 000: no load, dm_dout = 0
    - 001: lb, select addressed byte and sign-extend
    - 010: lbu, select addressed byte and zero-extend
    - 011: lh, select addressed half-word and sign-extend
    - 100: lhu, select addressed half-word and zero-extend
    - 101: lw, load full word

    dm_wr_ctrl encoding:
    - 00: no store
    - 01: sb, update one byte lane selected by dm_addr[1:0]
    - 10: sh, update one half-word lane selected by dm_addr[1]
    - 11: sw, update the full word

    Use little-endian byte lanes. For stores, preserve all byte lanes that are
    not selected by dm_wr_ctrl and dm_addr. Store enable is active whenever
    dm_wr_ctrl is not 00.
    """
    pass


@module
def ctrl(inst: In[32]) -> {
    "rf_wr_en": Out[1],
    "rf_wr_sel": Out[2],
    "do_jump": Out[1],
    "BrType": Out[3],
    "alu_a_sel": Out[1],
    "alu_b_sel": Out[1],
    "alu_ctrl": Out[4],
    "dm_rd_ctrl": Out[3],
    "dm_wr_ctrl": Out[2],
}:
    """Decode RV32I control signals for the CPU.

    Support LUI, AUIPC, JAL, JALR, BEQ, BNE, BLT, BGE, BLTU, BGEU,
    LB, LH, LW, LBU, LHU, SB, SH, SW, ADDI, SLTI, SLTIU, XORI, ORI, ANDI,
    SLLI, SRLI, SRAI, ADD, SUB, SLL, SLT, SLTU, XOR, SRL, SRA, OR, AND.

    rf_wr_en is 1 for instructions that write rd.
    rf_wr_sel encoding:
    - 00: write immediate, used by LUI
    - 01: write pc + 4, used by JAL and JALR
    - 10: write ALU result
    - 11: write data memory load result

    do_jump is 1 for JAL and JALR.
    BrType uses branch module encoding.
    alu_a_sel is 1 to select rs1, 0 to select pc.
    alu_b_sel is 1 to select immediate, 0 to select rs2.
    alu_ctrl uses alu module encoding.
    dm_rd_ctrl and dm_wr_ctrl use mem module encodings.
    """
    pass


@module
def CPU(clk: Clock, rst: In[1]) -> {}:
    reg_file0 = reg_file(
        clk=clk,
        rst=rst,
        A1="rs1",
        A2="rs2",
        A3="rd",
        WD="rf_wd",
        WE="rf_wr_en",
    )
    pc0 = pc(clk=clk, rst=rst, JUMP="JUMP", JUMP_PC="jump_pc")
    imm0 = imm("inst")
    branch0 = branch(REG1=reg_file0.RD1, REG2=reg_file0.RD2, Type="comp_ctrl")
    alu0 = alu(SrcA="alu_a", SrcB="alu_b", func="alu_ctrl")
    mem0 = mem(
        clk=clk,
        im_addr=pc0,
        dm_rd_ctrl="dm_rd_ctrl",
        dm_wr_ctrl="dm_wr_ctrl",
        dm_addr=alu0,
        dm_din=reg_file0.RD2,
    )
    ctrl0 = ctrl("inst")
    return f"""
    Build the CPU top level by wiring the named submodule instances.

    Define rs1 = inst[19:15], rs2 = inst[24:20], rd = inst[11:7].
    inst is {mem0.im_dout}.
    rf_wr_en is {ctrl0.rf_wr_en}.
    comp_ctrl is {ctrl0.BrType}.
    alu_ctrl is {ctrl0.alu_ctrl}.
    dm_rd_ctrl is {ctrl0.dm_rd_ctrl}.
    dm_wr_ctrl is {ctrl0.dm_wr_ctrl}.

    Define pc_plus4 = {pc0} + 4.
    JUMP is {branch0} OR {ctrl0.do_jump}.
    alu_a is {reg_file0.RD1} when {ctrl0.alu_a_sel} is 1, otherwise {pc0}.
    alu_b is {imm0} when {ctrl0.alu_b_sel} is 1, otherwise {reg_file0.RD2}.

    For JALR, jump_pc is {alu0} with bit 0 cleared.
    For other branch or jump instructions, jump_pc is {alu0}.

    rf_wd is selected by {ctrl0.rf_wr_sel}:
    - 00: {imm0}
    - 01: pc_plus4
    - 10: {alu0}
    - 11: {mem0.dm_dout}

    The module has no output ports, so finish with an output operation with empty args.
    """


design = Design()
design.add(CPU)


if __name__ == "__main__":
    cpu_sv = design.to_verilog(top="CPU")
    with open("cpu.sv", "w") as f:
        f.write(cpu_sv)
