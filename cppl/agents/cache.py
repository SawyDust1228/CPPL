"""Content-addressed cache for validated module JSON-IR."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

from .context import IR_VERSION, PROMPT_VERSION
from .models import CompiledModuleArtifact, ResolvedAgentConfig
from ..frontend.module import ModuleDef
from ..frontend.patterns import pattern_as_dict
from ..ir.errors import CircuitPPLError


class ModuleCache:
    def __init__(self, config: ResolvedAgentConfig, model_identity: dict[str, Any]):
        self.enabled = config.cache_enabled
        self.cache_dir = config.cache_dir
        self._config_identity = config.cache_identity()
        self._model_identity = model_identity

    @staticmethod
    def build_module_contract(mod: ModuleDef) -> dict[str, Any]:
        return {
            "name": mod.name,
            "ports": [
                {
                    "name": port.name,
                    "width": port.width,
                    "direction": port.direction,
                    "type": port.kind,
                }
                for port in mod.ports
            ],
            "description": mod.docstring,
            "patterns": [pattern_as_dict(pattern) for pattern in mod.patterns],
            "instances": [
                {
                    "module": inst.target_name,
                    "name": inst.name,
                    "args": inst.input_map,
                    "outputs": inst.output_ids,
                    "ports": [
                        {
                            "name": port.name,
                            "width": port.width,
                            "direction": port.direction,
                            "type": port.kind,
                        }
                        for port in inst.target_ports
                    ],
                }
                for inst in mod.instances
            ],
        }

    def key_for(
        self,
        mod: ModuleDef,
        dependency_modules: list[dict] | None = None,
    ) -> str:
        payload = {
            "ir_version": IR_VERSION,
            "prompt_version": PROMPT_VERSION,
            "module": self.build_module_contract(mod),
            "dependency_modules": dependency_modules or [],
            "model": self._model_identity,
            "config": self._config_identity,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def interface_hash(module_dict: dict) -> str:
        encoded = json.dumps(
            module_dict.get("ports", {}), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def artifact_for(
        cls,
        module_dict: dict,
        dependency_artifacts: list[CompiledModuleArtifact] | None = None,
        *,
        patterns_checked: int = 0,
    ) -> CompiledModuleArtifact:
        dependency_hashes = tuple(
            sorted(
                (artifact.module_name, artifact.semantic_hash)
                for artifact in dependency_artifacts or []
            )
        )
        encoded = json.dumps(
            {
                "module": module_dict,
                "dependency_hashes": dependency_hashes,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return CompiledModuleArtifact(
            module_name=module_dict["name"],
            semantic_hash=hashlib.sha256(encoded).hexdigest(),
            interface_hash=cls.interface_hash(module_dict),
            dependency_hashes=dependency_hashes,
            patterns_checked=patterns_checked,
            module_dict=module_dict,
        )

    def cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def load(
        self,
        key: str,
        validate: Callable[[dict], None],
    ) -> Optional[dict]:
        if not self.enabled:
            return None
        path = self.cache_path(key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("key") != key:
                return None
            module_dict = payload.get("module")
            if not isinstance(module_dict, dict):
                return None
            validate(module_dict)
            return module_dict
        except (OSError, ValueError, TypeError, CircuitPPLError):
            return None

    def store(self, key: str, module_dict: dict) -> None:
        if not self.enabled:
            return
        path = self.cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "key": key,
            "ir_version": IR_VERSION,
            "prompt_version": PROMPT_VERSION,
            "module": module_dict,
        }
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{key}.",
            suffix=".tmp",
            dir=path.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, path)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
