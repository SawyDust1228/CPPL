"""Tests for cppl.module_def: @module decorator and ModuleDef extraction."""

import pytest

from cppl.frontend.types import In, Out
from cppl.frontend.module import module, ModuleDef, PortInfo, InstanceCall


class TestModuleDecorator:
    """Test that @module correctly extracts ports and docstring."""

    def test_single_output(self):
        @module
        def Adder(a: In[8], b: In[8]) -> Out[8]:
            """out = a + b."""
            pass

        assert isinstance(Adder, ModuleDef)
        assert Adder.name == "Adder"
        assert Adder.docstring == "out = a + b."

        # 2 inputs + 1 output
        assert len(Adder.ports) == 3

        inputs = [p for p in Adder.ports if p.direction == "input"]
        outputs = [p for p in Adder.ports if p.direction == "output"]

        assert len(inputs) == 2
        assert inputs[0] == PortInfo("a", 8, "input")
        assert inputs[1] == PortInfo("b", 8, "input")

        assert len(outputs) == 1
        assert outputs[0] == PortInfo("out", 8, "output")

    def test_multi_output(self):
        @module
        def ALU(a: In[8], b: In[8]) -> {"sum": Out[8], "carry": Out[1]}:
            """Compute sum and carry."""
            pass

        outputs = [p for p in ALU.ports if p.direction == "output"]
        assert len(outputs) == 2
        assert outputs[0] == PortInfo("sum", 8, "output")
        assert outputs[1] == PortInfo("carry", 1, "output")

    def test_preserves_input_order(self):
        @module
        def M(x: In[1], y: In[2], z: In[4]) -> Out[1]:
            """Logic."""
            pass

        inputs = [p for p in M.ports if p.direction == "input"]
        assert [p.name for p in inputs] == ["x", "y", "z"]
        assert [p.width for p in inputs] == [1, 2, 4]

    def test_docstring_multiline(self):
        @module
        def M(a: In[1]) -> Out[1]:
            """
            Line 1.
            Line 2.
            """
            pass

        assert "Line 1." in M.docstring
        assert "Line 2." in M.docstring

    def test_func_reference_preserved(self):
        def orig(a: In[8]) -> Out[8]:
            """doc."""
            pass

        md = module(orig)
        assert md.func is orig


class TestModuleErrors:
    """Test that @module raises on invalid signatures."""

    def test_no_return_annotation(self):
        with pytest.raises(TypeError, match="return type annotation"):
            @module
            def Bad(a: In[8]):
                """doc."""
                pass

    def test_no_docstring(self):
        with pytest.raises(ValueError, match="non-empty docstring"):
            @module
            def Bad(a: In[8]) -> Out[8]:
                pass

    def test_empty_docstring(self):
        with pytest.raises(ValueError, match="non-empty docstring"):
            @module
            def Bad(a: In[8]) -> Out[8]:
                """"""
                pass

    def test_param_not_port_type(self):
        with pytest.raises(TypeError, match="must be annotated with In"):
            @module
            def Bad(a: int) -> Out[8]:
                """doc."""
                pass

    def test_param_is_output_type(self):
        with pytest.raises(TypeError, match="must be an input port"):
            @module
            def Bad(a: Out[8]) -> Out[8]:
                """doc."""
                pass

    def test_return_is_input_type(self):
        with pytest.raises(TypeError, match="must be an output port"):
            @module
            def Bad(a: In[8]) -> In[8]:
                """doc."""
                pass

    def test_multi_output_contains_input(self):
        with pytest.raises(TypeError, match="must be an output port"):
            @module
            def Bad(a: In[8]) -> {"x": In[8]}:
                """doc."""
                pass

    def test_return_unsupported_type(self):
        with pytest.raises(TypeError, match="must be Out"):
            @module
            def Bad(a: In[8]) -> int:
                """doc."""
                pass


class TestModuleInstantiation:
    """Test module instantiation via function-call syntax."""

    def _make_adder8(self):
        @module
        def Adder8(a: In[8], b: In[8]) -> Out[8]:
            """out equals a plus b."""
            pass
        return Adder8

    def test_single_output_instance(self):
        Adder8 = self._make_adder8()

        @module
        def Top(x: In[8], y: In[8]) -> Out[8]:
            result = Adder8(x, y)
            return f"out = {result}"

        assert len(Top.instances) == 1
        inst = Top.instances[0]
        assert inst.target_name == "Adder8"
        assert inst.input_map == {"a": "x", "b": "y"}
        assert inst.output_ids == ["adder8_out"]

    def test_multi_output_instance(self):
        @module
        def DualOut(a: In[4]) -> {"x": Out[4], "y": Out[2]}:
            """Produce x and y."""
            pass

        @module
        def Consumer(d: In[4]) -> Out[4]:
            result = DualOut(d)
            return f"out = {result.x}"

        assert len(Consumer.instances) == 1
        inst = Consumer.instances[0]
        assert inst.target_name == "DualOut"
        assert inst.output_ids == ["dualout_x", "dualout_y"]
        assert inst.input_map == {"a": "d"}

    def test_multiple_instances(self):
        Adder8 = self._make_adder8()

        @module
        def Top(a: In[8], b: In[8], c: In[8]) -> Out[8]:
            r1 = Adder8(a, b)
            r2 = Adder8(r1, c)
            return f"out = {r2}"

        assert len(Top.instances) == 2
        # first instance: no suffix
        assert Top.instances[0].output_ids == ["adder8_out"]
        # second instance: _1 suffix
        assert Top.instances[1].output_ids == ["adder8_out_1"]
        # second instance uses first output as input
        assert Top.instances[1].input_map["a"] == "adder8_out"

    def test_no_instances(self):
        @module
        def Simple(a: In[8]) -> Out[8]:
            """out = a."""
            pass

        assert Simple.instances == []

    def test_call_outside_module_body_raises(self):
        Adder8 = self._make_adder8()

        with pytest.raises(RuntimeError, match="outside of a @module"):
            Adder8("x", "y")

    def test_fstring_description(self):
        Adder8 = self._make_adder8()

        @module
        def Top(x: In[8], y: In[8]) -> Out[8]:
            result = Adder8(x, y)
            return f"out = {result} (added)"

        assert "adder8_out" in Top.docstring
        assert "(added)" in Top.docstring
