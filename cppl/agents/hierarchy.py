"""Deterministic hierarchy analysis for atomic CPPL module compilation."""

from __future__ import annotations

from dataclasses import dataclass

from ..frontend.module import ModuleDef


@dataclass(frozen=True)
class HierarchyPlan:
    """A dependency DAG whose nodes are exactly the user's ``@module`` objects."""

    modules: tuple[ModuleDef, ...]
    by_name: dict[str, ModuleDef]
    dependencies: dict[str, frozenset[str]]
    dependents: dict[str, frozenset[str]]
    depths: dict[str, int]


def build_hierarchy_plan(modules: list[ModuleDef]) -> HierarchyPlan:
    by_name = {module.name: module for module in modules}
    if len(by_name) != len(modules):
        raise ValueError("Duplicate ModuleDef names are not supported")

    known = set(by_name)
    dependencies = {
        module.name: frozenset(
            instance.target_name
            for instance in module.instances
            if instance.target_name in known
        )
        for module in modules
    }
    dependents_mutable = {name: set() for name in by_name}
    for name, required in dependencies.items():
        for dependency in required:
            dependents_mutable[dependency].add(name)

    visiting: list[str] = []
    visited: set[str] = set()
    depths: dict[str, int] = {}

    def visit(name: str) -> int:
        if name in depths:
            return depths[name]
        if name in visiting:
            start = visiting.index(name)
            cycle = visiting[start:] + [name]
            raise ValueError("Module dependency graph contains a cycle: " + " -> ".join(cycle))
        if name in visited:
            return depths[name]
        visiting.append(name)
        required_depths = [visit(dependency) for dependency in dependencies[name]]
        visiting.pop()
        visited.add(name)
        depths[name] = 0 if not required_depths else max(required_depths) + 1
        return depths[name]

    for module in modules:
        visit(module.name)

    return HierarchyPlan(
        modules=tuple(modules),
        by_name=by_name,
        dependencies=dependencies,
        dependents={
            name: frozenset(values) for name, values in dependents_mutable.items()
        },
        depths=depths,
    )


def collect_dependency_closure(name: str, plan: HierarchyPlan) -> set[str]:
    closure: set[str] = set()
    pending = list(plan.dependencies[name])
    while pending:
        dependency = pending.pop()
        if dependency in closure:
            continue
        closure.add(dependency)
        pending.extend(plan.dependencies[dependency])
    return closure


def collect_blocked_modules(failed: set[str], plan: HierarchyPlan) -> set[str]:
    blocked: set[str] = set()
    pending = list(failed)
    while pending:
        current = pending.pop()
        for dependent in plan.dependents.get(current, ()):
            if dependent not in failed and dependent not in blocked:
                blocked.add(dependent)
                pending.append(dependent)
    return blocked


__all__ = [
    "HierarchyPlan",
    "build_hierarchy_plan",
    "collect_blocked_modules",
    "collect_dependency_closure",
]
