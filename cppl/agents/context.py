"""Deterministic context construction and budgeting for module agents."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable

from .models import ContextBudgetError, DiagnosticPacket, ResolvedAgentConfig
from ..frontend.module import InstanceCall, ModuleDef
from ..frontend.patterns import pattern_as_dict


PROMPT_VERSION = "cppl-agent-v2.1-patterns"
IR_VERSION = "json-ir-v1"

COMPACT_SYSTEM_PROMPT = r"""You compile one hardware module into a CPPL JSON-IR body.
Return only one JSON array. The first character must be [ and the last must be ].
Inputs are existing SSA IDs. Every result ID is unique. End with exactly one output op.

Operations:
constant{id,op,value,width}; unary not/neg/reverse/or_reduce/and_reduce/xor_reduce{id,op,args:[x]};
binary add/sub/mul/div/div_s/mod_u/mod_s/and/or/xor/shl/shr_u/shr_s/eq/ne/lt_s/lt_u/le_s/le_u/gt_s/gt_u/ge_s/ge_u{id,op,args:[a,b]};
concat{id,op,args:[msb,...,lsb]}; extract{id,op,args:[x],lowBit,width};
mux{id,op,args:[sel,true,false]}; sext/zext{id,op,args:[x],width};
reg{id,op,args:[data],clock,reset?,resetValue?,enable?,width?};
mem{id:[read_ids],op,width,depth,clock,reset?,name?,initFile?,initFormat?,reads:[{addr,enable}],writes:[{addr,data,enable}]};
instance{id:[output_ids],op,module,name?,args:{child_input:value_id}};
output{op:"output",args:{output_port:value_id}}.

Binary operands must have equal widths; comparisons and reductions return 1 bit; mux select is 1 bit; concat is MSB to LSB. Decimal JSON numbers may be used in operand positions and will be converted to typed constants. Preserve every required instance exactly. Do not emit comments, markdown, prose, or a module wrapper."""

COMPRESSION_SYSTEM_PROMPT = r"""Compress one source chunk of hardware requirements without changing or omitting any behavior, encoding, width, reset, timing, memory, or interface constraint. Return only JSON: {"sources":["Rxxx"],"requirements":["atomic requirement",...]}. Preserve the supplied source ID exactly. Do not invent requirements."""


@dataclass(frozen=True)
class AgentContext:
    system_prompt: str
    user_prompt: str
    input_tokens_estimate: int
    compression_events: tuple[str, ...]


def estimate_tokens(text: str) -> int:
    """Conservative tokenizer-free estimate suitable across providers."""
    if not text:
        return 0
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    non_ascii_chars = len(text) - ascii_chars
    return max(1, (ascii_chars + 1) // 4 + non_ascii_chars)


def compact_description(description: str) -> tuple[str, list[str]]:
    events: list[str] = []
    paragraphs = re.split(r"\n\s*\n", description.strip())
    compacted: list[str] = []
    seen: set[str] = set()
    for paragraph in paragraphs:
        normalized_lines = [
            re.sub(r"[ \t]+", " ", line).strip()
            for line in paragraph.splitlines()
            if line.strip()
        ]
        normalized = "\n".join(normalized_lines)
        if not normalized:
            continue
        if normalized in seen:
            if "duplicate_requirements_removed" not in events:
                events.append("duplicate_requirements_removed")
            continue
        seen.add(normalized)
        compacted.append(normalized)
    result = "\n\n".join(compacted)
    if result != description.strip():
        events.append("description_whitespace_compacted")
    return result, events


def split_requirement_chunks(description: str, max_chars: int = 1600) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", description)
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        pieces = [paragraph]
        if len(paragraph) > max_chars:
            pieces = [
                paragraph[index : index + max_chars]
                for index in range(0, len(paragraph), max_chars)
            ]
        for piece in pieces:
            candidate = piece if not current else current + "\n\n" + piece
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = piece
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks


def requirement_chunks(
    description: str,
    config: ResolvedAgentConfig,
) -> list[tuple[str, str]]:
    """Split long requirements into independently budgeted source chunks."""
    available_chars = max(400, (config.input_budget_tokens - 300) * 3)
    chunks = split_requirement_chunks(description, max_chars=min(4000, available_chars))
    return [(f"R{index:03d}", chunk) for index, chunk in enumerate(chunks, start=1)]


def build_requirement_compression_context(
    source: str,
    text: str,
    config: ResolvedAgentConfig,
) -> AgentContext:
    payload = json.dumps(
        {"source": source, "text": text},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    estimate = estimate_tokens(COMPRESSION_SYSTEM_PROMPT) + estimate_tokens(payload)
    if estimate > config.input_budget_tokens:
        raise ContextBudgetError(
            f"Requirement source {source} needs approximately {estimate} input "
            f"tokens, but the input budget is {config.input_budget_tokens}."
        )
    return AgentContext(
        system_prompt=COMPRESSION_SYSTEM_PROMPT,
        user_prompt=payload,
        input_tokens_estimate=estimate,
        compression_events=("requirements_chunk_compressed",),
    )


def build_port_contract(mod: ModuleDef) -> list[dict]:
    return [
        {
            "name": port.name,
            "direction": port.direction,
            "width": port.width,
            "type": port.kind,
        }
        for port in mod.ports
    ]


def build_instance_contract(inst: InstanceCall, placement: str) -> dict:
    return {
        "placement": placement,
        "module": inst.target_name,
        "name": inst.name or None,
        "args": inst.input_map,
        "outputs": [
            {
                "id": output_id,
                "port": port.name,
                "width": port.width,
                "type": port.kind,
            }
            for port, output_id in zip(
                (p for p in inst.target_ports if p.direction == "output"),
                inst.output_ids,
            )
        ],
    }


def build_base_payload(
    mod: ModuleDef,
    preplaced_instances: Iterable[InstanceCall],
    deferred_instances: Iterable[InstanceCall],
    description: str,
) -> dict:
    return {
        "task": "generate_module_body",
        "module": mod.name,
        "ports": build_port_contract(mod),
        "requirements": description,
        "patterns": [pattern_as_dict(pattern) for pattern in mod.patterns],
        "instances": [
            *(
                build_instance_contract(inst, "already_preplaced")
                for inst in preplaced_instances
            ),
            *(
                build_instance_contract(inst, "must_emit")
                for inst in deferred_instances
            ),
        ],
        "rules": [
            "Do not emit already_preplaced instance operations.",
            "Emit every must_emit instance exactly once after its inputs exist.",
            "Drive every output port in the final output operation.",
            "Satisfy every executable input/output pattern.",
            "Each case starts from zero/reset simulation state.",
            "Each sequence starts from zero/reset state; its steps share state.",
            "Step inputs are partial updates and step outputs are partial checks.",
        ],
    }


def build_agent_context(
    mod: ModuleDef,
    preplaced_instances: list[InstanceCall],
    deferred_instances: list[InstanceCall],
    config: ResolvedAgentConfig,
    *,
    candidate: list | None = None,
    diagnostic: DiagnosticPacket | None = None,
    description_override: str | None = None,
) -> AgentContext:
    description, events = compact_description(
        description_override if description_override is not None else mod.docstring
    )
    payload = build_base_payload(
        mod,
        preplaced_instances,
        deferred_instances,
        description,
    )
    if description_override is not None:
        payload["rules"].append(
            "Implement every bracketed requirement source [Rxxx]; do not omit a source."
        )
    if candidate is not None:
        payload["task"] = "repair_module_body"
        payload["candidate"] = candidate
        payload["diagnostic"] = diagnostic.as_dict() if diagnostic else {}
        payload["rules"].append("Replace the candidate with a complete corrected body.")

    user_prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    estimate = estimate_tokens(COMPACT_SYSTEM_PROMPT) + estimate_tokens(user_prompt)

    if estimate > config.input_budget_tokens:
        raise ContextBudgetError(
            f"Module '{mod.name}' needs approximately {estimate} input tokens, "
            f"but the configured input budget is {config.input_budget_tokens}."
        )

    return AgentContext(
        system_prompt=COMPACT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        input_tokens_estimate=estimate,
        compression_events=tuple(dict.fromkeys(events)),
    )
