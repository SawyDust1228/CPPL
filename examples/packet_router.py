from cppl import module, In, Out, Design


@module
def HeaderDecoder(valid: In[1], dest: In[2]) -> {"to_lane0": Out[1], "to_lane1": Out[1], "drop": Out[1]}:
    """Decode a packet destination.

    A packet is accepted only when valid is 1.
    - dest 00 routes to lane0
    - dest 01 routes to lane1
    - dest 10 and 11 are dropped
    When valid is 0, both lane outputs and drop are 0.
    """
    pass


@module
def ParityCheck32(data: In[32], expected_odd: In[1]) -> {"ok": Out[1]}:
    """Check packet parity.

    Compute the xor reduction of all 32 data bits.
    ok is 1 when the computed parity equals expected_odd, otherwise 0.
    """
    pass


@module
def LaneGate32(data: In[32], route: In[1], parity_ok: In[1]) -> {"out_data": Out[32], "out_valid": Out[1]}:
    """Gate one output lane.

    out_valid is 1 only when route and parity_ok are both 1.
    When out_valid is 1, out_data equals data.
    When out_valid is 0, out_data is 0.
    """
    pass


@module
def PacketRouter2(
    in_valid: In[1],
    dest: In[2],
    parity: In[1],
    data: In[32],
) -> {
    "lane0_data": Out[32],
    "lane0_valid": Out[1],
    "lane1_data": Out[32],
    "lane1_valid": Out[1],
    "drop": Out[1],
}:
    decoded = HeaderDecoder(in_valid, dest)
    parity_ok = ParityCheck32(data, parity)
    lane0 = LaneGate32(data, decoded.to_lane0, parity_ok)
    lane1 = LaneGate32(data, decoded.to_lane1, parity_ok)
    return f"""
    Compose a two-lane packet router from the submodules.
    lane0_data is {lane0.out_data} and lane0_valid is {lane0.out_valid}.
    lane1_data is {lane1.out_data} and lane1_valid is {lane1.out_valid}.
    drop is 1 when {decoded.drop} is 1 or when in_valid is 1 and {parity_ok} is 0.
    """


design = Design()
design.add(PacketRouter2)

if __name__ == "__main__":
    print(design.to_verilog())
