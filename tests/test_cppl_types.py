"""Tests for cppl.types: In[N] and Out[N] port type descriptors."""

import pytest

from cppl.frontend.types import In, Out, _PortType


class TestIn:
    def test_basic(self):
        p = In[8]
        assert isinstance(p, _PortType)
        assert p.width == 8
        assert p.direction == "input"

    def test_width_1(self):
        p = In[1]
        assert p.width == 1
        assert p.direction == "input"

    def test_width_32(self):
        p = In[32]
        assert p.width == 32

    def test_repr(self):
        assert repr(In[8]) == "In[8]"

    def test_equality(self):
        assert In[8] == In[8]
        assert In[8] != In[16]
        assert In[8] != Out[8]

    def test_zero_width_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            In[0]

    def test_negative_width_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            In[-1]

    def test_non_int_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            In["8"]


class TestOut:
    def test_basic(self):
        p = Out[8]
        assert isinstance(p, _PortType)
        assert p.width == 8
        assert p.direction == "output"

    def test_width_1(self):
        p = Out[1]
        assert p.width == 1
        assert p.direction == "output"

    def test_repr(self):
        assert repr(Out[1]) == "Out[1]"

    def test_equality(self):
        assert Out[8] == Out[8]
        assert Out[8] != Out[16]
        assert Out[8] != In[8]

    def test_zero_width_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            Out[0]

    def test_negative_width_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            Out[-1]

    def test_non_int_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            Out[3.5]
