from pathlib import Path

from cppl import (
    AgentConfig,
    Case,
    Clock,
    CompileOptions,
    Design,
    In,
    MemoryFixture,
    Out,
    Sequence,
    Step,
    module,
)


PROBLEM_DIR = Path(__file__).resolve().parent / "problem"
INSTRUCTION_IMAGE = PROBLEM_DIR / "inst.dat"


def load_hex_words(path: Path) -> dict[int, int]:
    """Load the same whitespace/comment based hex format accepted by CPPL mem."""
    words: dict[int, int] = {}
    address = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("//", 1)[0].split("#", 1)[0]
        for token in line.split():
            if token.startswith("@"):
                address = int(token[1:].replace("_", ""), 16)
                continue
            words[address] = int(token.replace("_", ""), 16)
            address += 1
    return words


PROGRAM_WORDS = load_hex_words(INSTRUCTION_IMAGE)


def cpu_program_steps() -> list[Step]:
    """Clock the current problem image and check its architectural milestones."""
    checkpoints = {
        1: {"pc0.pc": 0x004, "reg_file0.reg_file[1]": 0x100},
        3: {"pc0.pc": 0x00C, "reg_file0.reg_file[3]": 0xFFFFFFFD},
        6: {"pc0.pc": 0x018, "reg_file0.reg_file[6]": 20},
        11: {
            "pc0.pc": 0x02C,
            "reg_file0.reg_file[9]": 28,
            "reg_file0.reg_file[10]": 28,
            "reg_file0.reg_file[11]": 0,
        },
        14: {
            "pc0.pc": 0x038,
            "reg_file0.reg_file[12]": 40,
            "reg_file0.reg_file[13]": 0x7FFFFFFE,
            "reg_file0.reg_file[14]": 0xFFFFFFFE,
        },
        16: {
            "pc0.pc": 0x040,
            "reg_file0.reg_file[15]": 8,
            "mem0.mem[64]": 8,
        },
        22: {
            "pc0.pc": 0x058,
            "reg_file0.reg_file[16]": 0xFFFFFFFD,
            "reg_file0.reg_file[17]": 0xFD,
            "reg_file0.reg_file[18]": 8,
            "reg_file0.reg_file[19]": 8,
            "mem0.mem[65]": 0x000800FD,
        },
        28: {
            "pc0.pc": 0x088,
            "reg_file0.reg_file[20]": 0,
            "reg_file0.reg_file[21]": 0,
            "reg_file0.reg_file[22]": 0,
            "reg_file0.reg_file[23]": 0,
            "reg_file0.reg_file[24]": 0,
            "reg_file0.reg_file[25]": 0,
        },
        29: {
            "pc0.pc": 0x090,
            "reg_file0.reg_file[26]": 0x08C,
            "reg_file0.reg_file[27]": 0,
        },
        31: {
            "pc0.pc": 0x0A4,
            "reg_file0.reg_file[28]": 0x09C,
            "reg_file0.reg_file[29]": 0x098,
            "reg_file0.reg_file[30]": 0,
            "reg_file0.reg_file[31]": 0,
            "reg_file0.reg_file[5]": 8,
        },
        36: {
            "pc0.pc": 0x0B4,
            "reg_file0.reg_file[0]": 0,
            "reg_file0.reg_file[20]": 2,
            "reg_file0.reg_file[21]": 3,
            "reg_file0.reg_file[22]": 4,
            "mem0.mem[66]": 2,
        },
    }
    steps = [
        Step(
            inputs={"clk": 0, "rst": 1},
            probes={"pc0.pc": 0, "reg_file0.reg_file[0]": 0},
        ),
        Step(inputs={"clk": 1}, probes={"pc0.pc": 0}),
        Step(inputs={"clk": 0, "rst": 0}),
    ]
    for executed_instruction in range(1, 37):
        steps.append(
            Step(
                inputs={"clk": 1},
                probes=checkpoints.get(executed_instruction, {}),
            )
        )
        if executed_instruction != 36:
            steps.append(Step(inputs={"clk": 0}))
    return steps


@module(
    patterns=[
        Sequence(
            name="reset_write_and_x0",
            steps=[
                Step(
                    inputs={
                        "clk": 0,
                        "rst": 1,
                        "A1": 0,
                        "A2": 1,
                        "A3": 1,
                        "WD": 0x12345678,
                        "WE": 1,
                    },
                    outputs={"RD1": 0, "RD2": 0},
                ),
                Step(inputs={"clk": 1}, outputs={"RD1": 0, "RD2": 0}),
                Step(inputs={"clk": 0, "rst": 0}),
                Step(inputs={"clk": 1}, outputs={"RD1": 0, "RD2": 0x12345678}),
                Step(inputs={"clk": 0, "A3": 0, "WD": 0xFFFFFFFF}),
                Step(inputs={"clk": 1}, outputs={"RD1": 0, "RD2": 0x12345678}),
            ],
        )
    ]
)
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


@module(
    patterns=[
        Sequence(
            name="reset_increment_and_jump",
            steps=[
                Step(
                    inputs={"clk": 0, "rst": 1, "JUMP": 0, "JUMP_PC": 0},
                    outputs={"pc": 0},
                ),
                Step(inputs={"clk": 1}, outputs={"pc": 0}),
                Step(inputs={"clk": 0, "rst": 0}),
                Step(inputs={"clk": 1}, outputs={"pc": 4}),
                Step(inputs={"clk": 0, "JUMP": 1, "JUMP_PC": 0x100}),
                Step(inputs={"clk": 1}, outputs={"pc": 0x100}),
            ],
        )
    ]
)
def pc(clk: Clock, rst: In[1], JUMP: In[1], JUMP_PC: In[32]) -> {"pc": Out[32]}:
    """Implement the program counter register.

    On reset, pc becomes 0.
    Otherwise, on each rising edge of clk:
    - if JUMP is 1, pc becomes JUMP_PC
    - otherwise, pc becomes pc + 4
    """
    pass


@module(
    patterns=[
        Case(name="addi_negative", inputs={"inst": 0xFFC10093}, outputs={"out": 0xFFFFFFFC}),
        Case(name="store_negative", inputs={"inst": 0xFE312C23}, outputs={"out": 0xFFFFFFF8}),
        Case(name="branch_negative", inputs={"inst": 0xFE2088E3}, outputs={"out": 0xFFFFFFF0}),
        Case(name="lui_upper", inputs={"inst": 0xABCDE0B7}, outputs={"out": 0xABCDE000}),
        Case(name="jal_positive", inputs={"inst": 0x014000EF}, outputs={"out": 20}),
        Case(name="unsupported", inputs={"inst": 0}, outputs={"out": 0}),
    ]
)
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


@module(
    patterns=[
        Case(name="disabled", inputs={"REG1": 7, "REG2": 7, "Type": 0}, outputs={"BrE": 0}),
        Case(name="beq_taken", inputs={"REG1": 7, "REG2": 7, "Type": 1}, outputs={"BrE": 1}),
        Case(name="bne_taken", inputs={"REG1": 7, "REG2": 8, "Type": 2}, outputs={"BrE": 1}),
        Case(name="blt_signed", inputs={"REG1": 0xFFFFFFFF, "REG2": 1, "Type": 3}, outputs={"BrE": 1}),
        Case(name="bge_signed", inputs={"REG1": 1, "REG2": 0xFFFFFFFF, "Type": 4}, outputs={"BrE": 1}),
        Case(name="bltu_false", inputs={"REG1": 0xFFFFFFFF, "REG2": 1, "Type": 5}, outputs={"BrE": 0}),
        Case(name="bgeu_taken", inputs={"REG1": 0xFFFFFFFF, "REG2": 1, "Type": 6}, outputs={"BrE": 1}),
    ]
)
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


@module(
    patterns=[
        Case(name="add_wrap", inputs={"SrcA": 0xFFFFFFFF, "SrcB": 1, "func": 0x0}, outputs={"ALUout": 0}),
        Case(name="subtract", inputs={"SrcA": 9, "SrcB": 4, "func": 0x8}, outputs={"ALUout": 5}),
        Case(name="shift_left_masked", inputs={"SrcA": 3, "SrcB": 35, "func": 0x1}, outputs={"ALUout": 24}),
        Case(name="shift_right_logical", inputs={"SrcA": 0x80000000, "SrcB": 4, "func": 0x5}, outputs={"ALUout": 0x08000000}),
        Case(name="shift_right_arithmetic", inputs={"SrcA": 0x80000000, "SrcB": 4, "func": 0xD}, outputs={"ALUout": 0xF8000000}),
        Case(name="signed_less_than", inputs={"SrcA": 0xFFFFFFFF, "SrcB": 1, "func": 0x2}, outputs={"ALUout": 1}),
        Case(name="unsigned_less_than", inputs={"SrcA": 1, "SrcB": 0xFFFFFFFF, "func": 0x3}, outputs={"ALUout": 1}),
        Case(name="xor", inputs={"SrcA": 0xAA55AA55, "SrcB": 0x0F0F0F0F, "func": 0x4}, outputs={"ALUout": 0xA55AA55A}),
        Case(name="or", inputs={"SrcA": 0xA0000005, "SrcB": 0x050000A0, "func": 0x6}, outputs={"ALUout": 0xA50000A5}),
        Case(name="and", inputs={"SrcA": 0xFF00FF00, "SrcB": 0x0F0F0F0F, "func": 0x7}, outputs={"ALUout": 0x0F000F00}),
        Case(name="pass_b", inputs={"SrcA": 0, "SrcB": 0xDEADBEEF, "func": 0xE}, outputs={"ALUout": 0xDEADBEEF}),
        Case(name="default_zero", inputs={"SrcA": 1, "SrcB": 2, "func": 0xF}, outputs={"ALUout": 0}),
    ]
)
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

    The lt_s and lt_u comparison operations produce a 1-bit result. Zero-extend
    that result to 32 bits before selecting it as ALUout. Every mux true/false
    pair must have identical widths, and every func decoder condition used as a
    mux selector must be a separate 1-bit equality result.
    """
    pass


@module(
    patterns=[
        Case(
            name="instruction_and_word_load",
            inputs={
                "clk": 0,
                "im_addr": 0,
                "dm_rd_ctrl": 5,
                "dm_wr_ctrl": 0,
                "dm_addr": 0,
                "dm_din": 0,
            },
            outputs={"im_dout": 0x80FF7F01, "dm_dout": 0x80FF7F01},
            fixtures=[MemoryFixture("mem", {0: 0x80FF7F01})],
        ),
        Case(
            name="signed_byte_load",
            inputs={
                "clk": 0,
                "im_addr": 0,
                "dm_rd_ctrl": 1,
                "dm_wr_ctrl": 0,
                "dm_addr": 3,
                "dm_din": 0,
            },
            outputs={"dm_dout": 0xFFFFFF80},
            fixtures=[MemoryFixture("mem", {0: 0x80FF7F01})],
        ),
        Sequence(
            name="full_word_store",
            fixtures=[MemoryFixture("mem", {0: 0})],
            steps=[
                Step(
                    inputs={
                        "clk": 0,
                        "im_addr": 0,
                        "dm_rd_ctrl": 0,
                        "dm_wr_ctrl": 3,
                        "dm_addr": 0,
                        "dm_din": 0x12345678,
                    },
                    outputs={"dm_dout": 0},
                ),
                Step(
                    inputs={"clk": 1},
                    probes={"mem[0]": 0x12345678},
                ),
            ],
        ),
    ]
)
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
    4096 word entries. Initialize that storage from
    examples/python/problem/inst.dat using hexadecimal text. The instruction
    side reads the word indexed by
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
    dm_wr_ctrl is not 00. Use the memory write port's optional 32-bit mask for
    sb/sh preservation: mask bit 1 updates that stored bit and mask bit 0 keeps
    the old stored bit. Omit the mask only for an unconditional full-word write.
    """
    pass


@module(
    patterns=[
        Case(name="lui", inputs={"inst": 0xABCDE0B7}, outputs={"rf_wr_en": 1, "rf_wr_sel": 0}),
        Case(name="auipc", inputs={"inst": 0x12345097}, outputs={"rf_wr_en": 1, "rf_wr_sel": 2, "alu_a_sel": 0, "alu_b_sel": 1, "alu_ctrl": 0}),
        Case(name="jal", inputs={"inst": 0x014000EF}, outputs={"rf_wr_en": 1, "rf_wr_sel": 1, "do_jump": 1, "alu_a_sel": 0, "alu_b_sel": 1, "alu_ctrl": 0}),
        Case(name="jalr", inputs={"inst": 0x008100E7}, outputs={"rf_wr_en": 1, "rf_wr_sel": 1, "do_jump": 1, "alu_a_sel": 1, "alu_b_sel": 1, "alu_ctrl": 0}),
        Case(name="beq", inputs={"inst": 0xFE2088E3}, outputs={"rf_wr_en": 0, "do_jump": 0, "BrType": 1, "alu_a_sel": 0, "alu_b_sel": 1, "alu_ctrl": 0}),
        Case(name="bne", inputs={"inst": 0x00001063}, outputs={"rf_wr_en": 0, "BrType": 2}),
        Case(name="blt", inputs={"inst": 0x00004063}, outputs={"rf_wr_en": 0, "BrType": 3}),
        Case(name="bge", inputs={"inst": 0x00005063}, outputs={"rf_wr_en": 0, "BrType": 4}),
        Case(name="bltu", inputs={"inst": 0x0020E663}, outputs={"rf_wr_en": 0, "BrType": 5}),
        Case(name="bgeu", inputs={"inst": 0x00007063}, outputs={"rf_wr_en": 0, "BrType": 6}),
        Case(name="lw", inputs={"inst": 0x00C12083}, outputs={"rf_wr_en": 1, "rf_wr_sel": 3, "alu_a_sel": 1, "alu_b_sel": 1, "alu_ctrl": 0, "dm_rd_ctrl": 5, "dm_wr_ctrl": 0}),
        Case(name="sw", inputs={"inst": 0xFE312C23}, outputs={"rf_wr_en": 0, "alu_a_sel": 1, "alu_b_sel": 1, "alu_ctrl": 0, "dm_rd_ctrl": 0, "dm_wr_ctrl": 3}),
        Case(name="addi", inputs={"inst": 0xFFC10093}, outputs={"rf_wr_en": 1, "rf_wr_sel": 2, "alu_a_sel": 1, "alu_b_sel": 1, "alu_ctrl": 0}),
        Case(name="sub", inputs={"inst": 0x403100B3}, outputs={"rf_wr_en": 1, "rf_wr_sel": 2, "alu_a_sel": 1, "alu_b_sel": 0, "alu_ctrl": 8}),
        Case(name="srai", inputs={"inst": 0x40315093}, outputs={"rf_wr_en": 1, "rf_wr_sel": 2, "alu_a_sel": 1, "alu_b_sel": 1, "alu_ctrl": 13}),
        Case(name="and", inputs={"inst": 0x003170B3}, outputs={"rf_wr_en": 1, "rf_wr_sel": 2, "alu_a_sel": 1, "alu_b_sel": 0, "alu_ctrl": 7}),
    ]
)
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


@module(
    patterns=[
        Sequence(
            name="execute_problem_instruction_image",
            fixtures=[MemoryFixture("mem0.mem", PROGRAM_WORDS)],
            steps=cpu_program_steps(),
        )
    ]
)
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


design = Design(
    agent_config=AgentConfig(
        max_parallelism=2,
        fail_fast=False,
        request_timeout=300,
        transport_retries=4,
        output_tokens=8000,
        module_deadline_seconds=1800,
        max_module_tokens=300000,
        max_tool_rounds=6,
    )
)
design.add(CPU)


if __name__ == "__main__":
    cpu_sv = design.to_verilog(
        top="CPU",
        max_retries=5,
        options=CompileOptions(run_id="rv32i-cpu", resume=True),
    )
    with open("cpu.sv", "w") as f:
        f.write(cpu_sv)
