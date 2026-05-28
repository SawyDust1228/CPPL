"""@module decorator and ModuleDef dataclass for CPPL DSL."""

from __future__ import annotations

import threading
import ast
import inspect
import textwrap
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .types import _PortType


@dataclass
class PortInfo:
    """Describes a single port extracted from a function signature."""
    name: str
    width: int
    direction: str  # "input" | "output"
    kind: str = "bits"


@dataclass
class InstanceCall:
    """Records one module-instantiation captured during @module body tracing."""
    name: Optional[str]
    target_name: str
    target_ports: List[PortInfo]
    input_map: Dict[str, str]   # child_input_port → parent_value_id
    output_ids: List[str]       # parent value IDs for child outputs
    target_mod: "ModuleDef | None" = None  # reference to target ModuleDef


class _PortProxy:
    """Proxy for an input port name, returned during body tracing."""

    def __init__(self, name: str):
        self._name = name

    def __str__(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"_PortProxy({self._name!r})"


class _InstanceOutputProxy:
    """Proxy for a single instance output value ID."""

    def __init__(self, name: str):
        self._name = name

    def __str__(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"_InstanceOutputProxy({self._name!r})"


class _MultiInstanceOutputProxy:
    """Proxy for multi-output instance; attribute access yields per-port proxies."""

    def __init__(self, outputs: Dict[str, _InstanceOutputProxy]):
        object.__setattr__(self, "_outputs", outputs)

    def __getattr__(self, name: str) -> _InstanceOutputProxy:
        outputs = object.__getattribute__(self, "_outputs")
        if name in outputs:
            return outputs[name]
        raise AttributeError(
            f"Instance has no output port '{name}'. "
            f"Available: {list(outputs.keys())}"
        )

    def __str__(self) -> str:
        outputs = object.__getattribute__(self, "_outputs")
        return ", ".join(str(v) for v in outputs.values())

    def __repr__(self) -> str:
        outputs = object.__getattribute__(self, "_outputs")
        return f"_MultiInstanceOutputProxy({outputs!r})"


_capture_ctx = threading.local()


def _get_value_name(arg: object) -> str:
    """Extract the value-ID string from a proxy object or raise."""
    if isinstance(arg, (_PortProxy, _InstanceOutputProxy)):
        return arg._name
    if isinstance(arg, str):
        return arg
    raise TypeError(
        f"Instance argument must be a port proxy or instance output proxy, "
        f"got {type(arg).__name__}: {arg!r}"
    )


@dataclass
class ModuleDef:
    """All metadata captured from a @module-decorated function."""
    name: str
    ports: List[PortInfo]
    docstring: str
    func: Callable
    instances: List[InstanceCall] = field(default_factory=list)

    def __call__(self, *args, **kwargs):
        """Instantiate this module inside another @module body.

        Maps positional/keyword args to child input ports and records an
        InstanceCall in the active capture context.
        """
        calls = getattr(_capture_ctx, "calls", None)
        if calls is None:
            raise RuntimeError(
                f"Cannot call ModuleDef '{self.name}' outside of a @module "
                f"function body. Module instantiation is only valid inside "
                f"another @module-decorated function."
            )

        input_ports = [p for p in self.ports if p.direction == "input"]
        output_ports = [p for p in self.ports if p.direction == "output"]

        input_map: Dict[str, str] = {}

        for i, arg in enumerate(args):
            if i >= len(input_ports):
                raise TypeError(
                    f"Module '{self.name}' has {len(input_ports)} input ports, "
                    f"but {len(args)} positional arguments were given."
                )
            port_name = input_ports[i].name
            input_map[port_name] = _get_value_name(arg)

        input_port_names = {p.name for p in input_ports}
        for kw, arg in kwargs.items():
            if kw not in input_port_names:
                raise TypeError(
                    f"Module '{self.name}' has no input port '{kw}'. "
                    f"Available: {[p.name for p in input_ports]}"
                )
            if kw in input_map:
                raise TypeError(
                    f"Input port '{kw}' of module '{self.name}' specified "
                    f"both positionally and as keyword argument."
                )
            input_map[kw] = _get_value_name(arg)

        missing = [p.name for p in input_ports if p.name not in input_map]
        if missing:
            raise TypeError(
                f"Missing input port(s) for module '{self.name}': {missing}"
            )

        module_lower = self.name.lower()

        same_count = sum(
            1 for c in calls if c.target_name == self.name
        )
        suffix = f"_{same_count}" if same_count > 0 else ""

        output_ids: List[str] = []
        for p in output_ports:
            output_ids.append(f"{module_lower}_{p.name}{suffix}")

        inst = InstanceCall(
            name=None,
            target_name=self.name,
            target_ports=list(self.ports),
            input_map=input_map,
            output_ids=output_ids,
            target_mod=self,
        )
        calls.append(inst)

        if len(output_ports) == 1:
            return _InstanceOutputProxy(output_ids[0])
        else:
            outputs_dict = {}
            for p, oid in zip(output_ports, output_ids):
                outputs_dict[p.name] = _InstanceOutputProxy(oid)
            return _MultiInstanceOutputProxy(outputs_dict)


def _instance_assignment_names(func: Callable) -> List[Optional[str]]:
    """Return assignment targets for ModuleDef calls in source order."""
    try:
        source = inspect.getsource(func)
    except OSError:
        return []

    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return []

    module_names = {
        name
        for name, value in func.__globals__.items()
        if isinstance(value, ModuleDef)
    }

    names: List[Optional[str]] = []

    def _call_target(node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in module_names:
                return node.func.id
        return None

    class Visitor(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign) -> None:
            if _call_target(node.value):
                target = node.targets[0] if node.targets else None
                if isinstance(target, ast.Name):
                    names.append(target.id)
                else:
                    names.append(None)
                return
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if _call_target(node.value):
                if isinstance(node.target, ast.Name):
                    names.append(node.target.id)
                else:
                    names.append(None)
                return
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if _call_target(node):
                names.append(None)
                return
            self.generic_visit(node)

    Visitor().visit(tree)
    return names


def module(func: Callable) -> ModuleDef:
    """Decorator that converts a typed Python function into a :class:`ModuleDef`.

    Input ports come from function parameters annotated with ``In[N]`` or ``Clock``.
    Output ports come from the return annotation:
      - ``-> Out[N]``       : single output named ``"out"``
      - ``-> {"x": Out[N], ...}`` : named outputs
      - ``-> {}``                 : no output ports

    If the function body calls other ModuleDef objects (module instantiation),
    the calls are captured and the function may return an f-string description
    instead of using a docstring.
    """
    annotations = func.__annotations__
    ports: List[PortInfo] = []

    for param_name, ann in annotations.items():
        if param_name == "return":
            continue
        if not isinstance(ann, _PortType):
            raise TypeError(
                f"Parameter '{param_name}' of module '{func.__name__}' "
                f"must be annotated with In[N], got {ann!r}"
            )
        if ann.direction != "input":
            raise TypeError(
                f"Parameter '{param_name}' of module '{func.__name__}' "
                f"must be an input port (In[N]), got {ann!r}"
            )
        if ann.kind == "clock" and ann.width != 1:
            raise TypeError(
                f"Clock parameter '{param_name}' of module '{func.__name__}' "
                "must be 1 bit wide."
            )
        ports.append(
            PortInfo(
                name=param_name,
                width=ann.width,
                direction="input",
                kind=ann.kind,
            )
        )

    ret = annotations.get("return")
    if ret is None:
        raise TypeError(
            f"Module '{func.__name__}' must have a return type annotation "
            f"(Out[N] or dict of Out[N])."
        )

    if isinstance(ret, _PortType):
        if ret.direction != "output":
            raise TypeError(
                f"Return annotation of module '{func.__name__}' "
                f"must be an output port (Out[N]), got {ret!r}"
            )
        if ret.kind == "clock":
            raise TypeError(
                f"Return annotation of module '{func.__name__}' "
                "cannot be Clock; Clock is input-only."
            )
        ports.append(
            PortInfo(
                name="out",
                width=ret.width,
                direction="output",
                kind=ret.kind,
            )
        )
    elif isinstance(ret, dict):
        for port_name, ann in ret.items():
            if not isinstance(ann, _PortType):
                raise TypeError(
                    f"Output '{port_name}' of module '{func.__name__}' "
                    f"must be Out[N], got {ann!r}"
                )
            if ann.direction != "output":
                raise TypeError(
                    f"Output '{port_name}' of module '{func.__name__}' "
                    f"must be an output port (Out[N]), got {ann!r}"
                )
            if ann.kind == "clock":
                raise TypeError(
                    f"Output '{port_name}' of module '{func.__name__}' "
                    "cannot be Clock; Clock is input-only."
                )
            ports.append(
                PortInfo(
                    name=port_name,
                    width=ann.width,
                    direction="output",
                    kind=ann.kind,
                )
            )
    else:
        raise TypeError(
            f"Return annotation of module '{func.__name__}' must be Out[N] or "
            f"a dict of {{name: Out[N]}}, got {ret!r}"
        )

    _capture_ctx.calls = []
    try:
        input_ports = [p for p in ports if p.direction == "input"]
        proxy_kwargs = {p.name: _PortProxy(p.name) for p in input_ports}
        try:
            result = func(**proxy_kwargs)
        except Exception:
            result = None
        instances = list(_capture_ctx.calls)
    finally:
        _capture_ctx.calls = None

    for inst, name in zip(instances, _instance_assignment_names(func)):
        inst.name = name

    if isinstance(result, str) and result.strip():
        docstring = result.strip()
    else:
        docstring = (func.__doc__ or "").strip()

    if not docstring:
        raise ValueError(
            f"Module '{func.__name__}' must have a non-empty docstring "
            f"describing its logic."
        )

    return ModuleDef(
        name=func.__name__,
        ports=ports,
        docstring=docstring,
        func=func,
        instances=instances,
    )
