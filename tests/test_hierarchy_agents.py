import json
import threading
import time

from cppl import AgentConfig, Case, In, Out, module
from cppl.agents.backend import BackendResponse
from cppl.agents.cache import ModuleCache
from cppl.agents.hierarchy import build_hierarchy_plan
from cppl.agents.runtime import CompilationCoordinator


def config(tmp_path, *, parallelism=4, fail_fast=True):
    return AgentConfig(
        log_enabled=False,
        cache_enabled=False,
        checkpoint_enabled=False,
        cache_dir=tmp_path / "cache",
        checkpoint_path=tmp_path / "checkpoints.sqlite3",
        max_parallelism=parallelism,
        fail_fast=fail_fast,
        context_window_tokens=8192,
        output_tokens=2048,
        safety_margin_tokens=512,
    )


def test_large_multi_output_module_remains_one_atomic_task(tmp_path):
    @module
    def Decoder(inst: In[32]) -> {
        "a": Out[1],
        "b": Out[1],
        "c": Out[1],
        "d": Out[1],
    }:
        """Decode a deliberately detailed instruction control table.

        This long description previously caused output-cone planning. The user-defined
        module is now the only compilation boundary regardless of its complexity or
        output count. Reserved encodings drive zero on every output.
        """
        pass

    class Backend:
        calls = 0

        def __init__(self, resolved):
            pass

        def generate(self, system_prompt, user_prompt, *, output_tokens=None):
            self.__class__.calls += 1
            return BackendResponse(
                json.dumps(
                    [
                        {"id": "zero", "op": "constant", "value": 0, "width": 1},
                        {
                            "op": "output",
                            "args": {"a": "zero", "b": "zero", "c": "zero", "d": "zero"},
                        },
                    ]
                ),
                0,
            )

    report = CompilationCoordinator(
        config(tmp_path),
        backend_factory=Backend,
        model_identity={"model": "fake"},
    ).compile([Decoder])

    assert report.success
    assert Backend.calls == 1
    assert set(report.module_reports) == {"Decoder"}
    assert report.module_reports["Decoder"].plan_strategy == "hierarchy_node"
    assert report.module_reports["Decoder"].planned_blocks == []


def test_hierarchy_plan_uses_user_modules_as_dag_nodes():
    @module
    def Leaf(a: In[8]) -> Out[8]:
        """out equals a."""
        pass

    @module
    def Left(a: In[8]) -> Out[8]:
        leaf = Leaf(a)
        return f"out equals {leaf}."

    @module
    def Right(a: In[8]) -> Out[8]:
        leaf = Leaf(a)
        return f"out equals {leaf}."

    @module
    def Top(a: In[8]) -> Out[8]:
        left = Left(a)
        Right(a)
        return f"out equals {left}."

    plan = build_hierarchy_plan([Leaf, Left, Right, Top])
    assert plan.dependencies["Leaf"] == frozenset()
    assert plan.dependencies["Left"] == frozenset({"Leaf"})
    assert plan.dependencies["Right"] == frozenset({"Leaf"})
    assert plan.dependencies["Top"] == frozenset({"Left", "Right"})
    assert plan.depths == {"Leaf": 0, "Left": 1, "Right": 1, "Top": 2}


def test_ready_siblings_compile_in_parallel_and_parent_waits(tmp_path):
    @module
    def A(x: In[8]) -> Out[8]:
        """out equals x."""
        pass

    @module
    def B(x: In[8]) -> Out[8]:
        """out equals x."""
        pass

    @module
    def Top(x: In[8]) -> Out[8]:
        a = A(x)
        B(x)
        return f"out equals {a}."

    class Backend:
        lock = threading.Lock()
        active = 0
        max_active = 0
        leaf_finished = 0

        def __init__(self, resolved):
            pass

        def generate(self, system_prompt, user_prompt, *, output_tokens=None):
            module_name = json.loads(user_prompt)["module"]
            with self.__class__.lock:
                self.__class__.active += 1
                self.__class__.max_active = max(
                    self.__class__.max_active, self.__class__.active
                )
                if module_name == "Top":
                    assert self.__class__.leaf_finished == 2
            if module_name in {"A", "B"}:
                time.sleep(0.08)
                body = [{"op": "output", "args": {"out": "x"}}]
            else:
                body = [{"op": "output", "args": {"out": "a_out"}}]
            with self.__class__.lock:
                self.__class__.active -= 1
                if module_name in {"A", "B"}:
                    self.__class__.leaf_finished += 1
            return BackendResponse(json.dumps(body), 0)

    report = CompilationCoordinator(
        config(tmp_path, parallelism=2),
        backend_factory=Backend,
        model_identity={"model": "fake"},
    ).compile([A, B, Top])

    assert report.success
    assert Backend.max_active == 2
    assert report.module_reports["A"].hierarchy_depth == 0
    assert report.module_reports["Top"].hierarchy_depth == 1
    assert report.module_reports["Top"].direct_dependencies == ["A", "B"]


def test_artifact_hash_propagates_descendant_changes(tmp_path):
    cache = ModuleCache(config(tmp_path).resolve(), {"model": "fake"})
    leaf_v1 = cache.artifact_for(
        {"name": "Leaf", "ports": {}, "body": [{"op": "output", "args": {}}]}
    )
    leaf_v2 = cache.artifact_for(
        {
            "name": "Leaf",
            "ports": {},
            "body": [
                {"id": "x", "op": "constant", "value": 1, "width": 1},
                {"op": "output", "args": {}},
            ],
        }
    )
    middle_dict = {
        "name": "Middle",
        "ports": {},
        "body": [{"op": "output", "args": {}}],
    }
    middle_v1 = cache.artifact_for(middle_dict, [leaf_v1])
    middle_v2 = cache.artifact_for(middle_dict, [leaf_v2])
    assert middle_v1.semantic_hash != middle_v2.semantic_hash


def test_dependency_cycle_fails_before_llm(tmp_path):
    @module
    def A(x: In[8]) -> Out[8]:
        """out equals x."""
        pass

    @module
    def B(x: In[8]) -> Out[8]:
        """out equals x."""
        pass

    from cppl.frontend.module import InstanceCall

    A.instances.append(
        InstanceCall(None, "B", list(B.ports), {"x": "x"}, ["b_out"], B)
    )
    B.instances.append(
        InstanceCall(None, "A", list(A.ports), {"x": "x"}, ["a_out"], A)
    )

    class Backend:
        calls = 0

        def __init__(self, resolved):
            pass

        def generate(self, system_prompt, user_prompt, *, output_tokens=None):
            self.__class__.calls += 1
            raise AssertionError("LLM must not be called for an invalid hierarchy")

    report = CompilationCoordinator(
        config(tmp_path),
        backend_factory=Backend,
        model_identity={"model": "fake"},
    ).compile([A, B])

    assert not report.success
    assert "cycle" in report.design_error.lower()
    assert Backend.calls == 0


def test_failed_branch_blocks_only_its_ancestors_when_fail_fast_is_disabled(tmp_path):
    @module
    def Bad(x: In[8]) -> Out[8]:
        """out equals x."""
        pass

    @module
    def Good(x: In[8]) -> Out[8]:
        """out equals x."""
        pass

    @module
    def BadParent(x: In[8]) -> Out[8]:
        bad = Bad(x)
        return f"out equals {bad}."

    class Backend:
        def __init__(self, resolved):
            pass

        def generate(self, system_prompt, user_prompt, *, output_tokens=None):
            module_name = json.loads(user_prompt)["module"]
            if module_name == "Bad":
                return BackendResponse("not-json", 0)
            return BackendResponse(
                json.dumps([{"op": "output", "args": {"out": "x"}}]), 0
            )

    report = CompilationCoordinator(
        config(tmp_path, parallelism=2, fail_fast=False),
        backend_factory=Backend,
        model_identity={"model": "fake"},
    ).compile([Bad, Good, BadParent], max_retries=1)

    assert not report.success
    assert report.module_reports["Bad"].status == "failed"
    assert report.module_reports["Good"].status == "success"
    assert report.module_reports["BadParent"].status == "blocked"
    assert report.module_reports["BadParent"].failure_origin == "dependency"
