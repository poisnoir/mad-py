"""
test_mad.py — Unit tests for mad.py binary serialization library.

Run with:  python -m pytest test_mad.py -v
"""

import math
import struct
from dataclasses import dataclass
from typing import Annotated, Dict, List

import pytest

from mad import FixedArray, Mad

# =============================================================================
# Helpers
# =============================================================================


def roundtrip(codec: Mad, value):
    """Encode then decode; return the reconstructed value."""
    encoded = codec.encode(value)
    return codec.decode(encoded)


# =============================================================================
# Primitive: bool
# =============================================================================


class TestBool:
    codec = Mad(bool)

    def test_true_encodes_to_single_byte(self):
        assert self.codec.encode(True) == b"\x01"

    def test_false_encodes_to_single_byte(self):
        assert self.codec.encode(False) == b"\x00"

    def test_roundtrip_true(self):
        assert roundtrip(self.codec, True) is True

    def test_roundtrip_false(self):
        assert roundtrip(self.codec, False) is False

    def test_size_is_one(self):
        assert self.codec.get_required_size(True) == 1
        assert self.codec.get_required_size(False) == 1

    def test_code(self):
        assert self.codec.code == "0"


# =============================================================================
# Primitives: annotated integers
# =============================================================================


@dataclass
class _Int8Wrapper:
    v: Annotated[int, "int8"]


@dataclass
class _Uint8Wrapper:
    v: Annotated[int, "uint8"]


@dataclass
class _Int16Wrapper:
    v: Annotated[int, "int16"]


@dataclass
class _Uint16Wrapper:
    v: Annotated[int, "uint16"]


@dataclass
class _Int32Wrapper:
    v: Annotated[int, "int32"]


@dataclass
class _Uint32Wrapper:
    v: Annotated[int, "uint32"]


@dataclass
class _Int64Wrapper:
    v: Annotated[int, "int64"]


@dataclass
class _Uint64Wrapper:
    v: Annotated[int, "uint64"]


class TestAnnotatedIntegers:
    def test_int8_positive(self):
        c = Mad(_Int8Wrapper)
        assert roundtrip(c, _Int8Wrapper(127)).v == 127

    def test_int8_negative(self):
        c = Mad(_Int8Wrapper)
        assert roundtrip(c, _Int8Wrapper(-1)).v == -1

    def test_int8_min(self):
        c = Mad(_Int8Wrapper)
        assert roundtrip(c, _Int8Wrapper(-128)).v == -128

    def test_uint8_max(self):
        c = Mad(_Uint8Wrapper)
        assert roundtrip(c, _Uint8Wrapper(255)).v == 255

    def test_uint8_zero(self):
        c = Mad(_Uint8Wrapper)
        assert roundtrip(c, _Uint8Wrapper(0)).v == 0

    def test_int16_roundtrip(self):
        c = Mad(_Int16Wrapper)
        for v in (-32768, -1, 0, 1, 32767):
            assert roundtrip(c, _Int16Wrapper(v)).v == v

    def test_uint16_max(self):
        c = Mad(_Uint16Wrapper)
        assert roundtrip(c, _Uint16Wrapper(65535)).v == 65535

    def test_int32_roundtrip(self):
        c = Mad(_Int32Wrapper)
        for v in (-(2**31), -1, 0, 1, 2**31 - 1):
            assert roundtrip(c, _Int32Wrapper(v)).v == v

    def test_uint32_max(self):
        c = Mad(_Uint32Wrapper)
        assert roundtrip(c, _Uint32Wrapper(2**32 - 1)).v == 2**32 - 1

    def test_int64_roundtrip(self):
        c = Mad(_Int64Wrapper)
        for v in (-(2**63), -1, 0, 1, 2**63 - 1):
            assert roundtrip(c, _Int64Wrapper(v)).v == v

    def test_uint64_max(self):
        c = Mad(_Uint64Wrapper)
        assert roundtrip(c, _Uint64Wrapper(2**64 - 1)).v == 2**64 - 1

    def test_byte_sizes(self):
        assert Mad(_Int8Wrapper).get_required_size(_Int8Wrapper(0)) == 1
        assert Mad(_Int16Wrapper).get_required_size(_Int16Wrapper(0)) == 2
        assert Mad(_Int32Wrapper).get_required_size(_Int32Wrapper(0)) == 4
        assert Mad(_Int64Wrapper).get_required_size(_Int64Wrapper(0)) == 8

    def test_big_endian_byte_order_int16(self):
        c = Mad(_Int16Wrapper)
        # 256 = 0x0100 in big-endian
        raw = c.encode(_Int16Wrapper(256))
        assert raw == b"\x01\x00"

    def test_big_endian_byte_order_int32(self):
        c = Mad(_Int32Wrapper)
        raw = c.encode(_Int32Wrapper(0x01020304))
        assert raw == b"\x01\x02\x03\x04"


# =============================================================================
# Primitives: plain int / float (default widths)
# =============================================================================


class TestDefaultInt:
    codec = Mad(int)

    def test_roundtrip_positive(self):
        assert roundtrip(self.codec, 42) == 42

    def test_roundtrip_negative(self):
        assert roundtrip(self.codec, -100) == -100

    def test_roundtrip_zero(self):
        assert roundtrip(self.codec, 0) == 0

    def test_size_is_eight(self):
        assert self.codec.get_required_size(0) == 8

    def test_code_is_three(self):
        assert self.codec.code == "3"

    def test_default_is_int64(self):
        # Big-endian 8-byte representation of 1
        assert self.codec.encode(1) == b"\x00\x00\x00\x00\x00\x00\x00\x01"


class TestDefaultFloat:
    codec = Mad(float)

    def test_roundtrip(self):
        assert roundtrip(self.codec, 3.14159) == 3.14159

    def test_roundtrip_negative(self):
        assert roundtrip(self.codec, -2.718) == -2.718

    def test_roundtrip_zero(self):
        assert roundtrip(self.codec, 0.0) == 0.0

    def test_size_is_eight(self):
        assert self.codec.get_required_size(0.0) == 8

    def test_infinity(self):
        assert roundtrip(self.codec, math.inf) == math.inf

    def test_negative_infinity(self):
        assert roundtrip(self.codec, -math.inf) == -math.inf

    def test_nan(self):
        assert math.isnan(roundtrip(self.codec, math.nan))


# =============================================================================
# Primitives: float32 annotation
# =============================================================================


@dataclass
class _Float32Wrapper:
    v: Annotated[float, "float32"]


class TestFloat32:
    def test_roundtrip_approximate(self):
        c = Mad(_Float32Wrapper)
        dec = roundtrip(c, _Float32Wrapper(3.14))
        assert math.isclose(dec.v, 3.14, rel_tol=1e-5)

    def test_size_is_four(self):
        c = Mad(_Float32Wrapper)
        assert c.get_required_size(_Float32Wrapper(0.0)) == 4


# =============================================================================
# String
# =============================================================================


class TestString:
    codec = Mad(str)

    def test_empty_string(self):
        assert roundtrip(self.codec, "") == ""

    def test_empty_string_encodes_to_four_zero_bytes(self):
        assert self.codec.encode("") == b"\x00\x00\x00\x00"

    def test_ascii_string(self):
        assert roundtrip(self.codec, "hello") == "hello"

    def test_unicode_string(self):
        assert roundtrip(self.codec, "héllo") == "héllo"

    def test_unicode_cjk(self):
        assert roundtrip(self.codec, "日本語") == "日本語"

    def test_emoji(self):
        assert roundtrip(self.codec, "hi 🎉") == "hi 🎉"

    def test_length_prefix_is_byte_length_not_char_length(self):
        # "é" is 2 bytes in UTF-8
        encoded = self.codec.encode("é")
        n = struct.unpack(">I", encoded[:4])[0]
        assert n == 2

    def test_size_is_four_plus_utf8_bytes(self):
        assert self.codec.get_required_size("hello") == 9  # 4 + 5
        assert self.codec.get_required_size("é") == 6  # 4 + 2
        assert self.codec.get_required_size("") == 4

    def test_code(self):
        assert self.codec.code == "4"

    def test_long_string_roundtrip(self):
        s = "x" * 10_000
        assert roundtrip(self.codec, s) == s

    def test_decode_truncated_buffer_raises(self):
        with pytest.raises(ValueError, match="[Bb]uffer"):
            self.codec.decode(b"\x00\x00\x00\x05he")  # claims 5 bytes, only 2 available


# =============================================================================
# Dataclass / struct
# =============================================================================


@dataclass
class Point:
    x: float
    y: float


@dataclass
class NamedPoint:
    name: str
    x: float
    y: float


@dataclass
class Inner:
    value: int


@dataclass
class Outer:
    inner: Inner
    label: str


@dataclass
class FieldOrder:
    zebra: int
    apple: int
    mango: int


class TestStruct:
    def test_simple_point_roundtrip(self):
        c = Mad(Point)
        assert roundtrip(c, Point(1.0, 2.0)) == Point(1.0, 2.0)

    def test_point_with_negatives(self):
        c = Mad(Point)
        assert roundtrip(c, Point(-3.5, 0.0)) == Point(-3.5, 0.0)

    def test_named_point_roundtrip(self):
        c = Mad(NamedPoint)
        v = NamedPoint("origin", 0.0, 0.0)
        assert roundtrip(c, v) == v

    def test_nested_dataclass_roundtrip(self):
        c = Mad(Outer)
        v = Outer(inner=Inner(42), label="hello")
        assert roundtrip(c, v) == v

    def test_fields_sorted_alphabetically_in_encoding(self):
        # Encoding order must be alphabetical by field name (apple < mango < zebra).
        c = Mad(FieldOrder)
        v = FieldOrder(zebra=1, apple=2, mango=3)
        raw = c.encode(v)
        # Each int64 is 8 bytes; order should be apple=2, mango=3, zebra=1
        apple, mango, zebra = struct.unpack(">qqq", raw)
        assert (apple, mango, zebra) == (2, 3, 1)

    def test_decode_respects_alphabetical_order(self):
        c = Mad(FieldOrder)
        v = FieldOrder(zebra=10, apple=20, mango=30)
        assert roundtrip(c, v) == v

    def test_size_is_sum_of_fields(self):
        c = Mad(Point)
        assert c.get_required_size(Point(0.0, 0.0)) == 16  # 8 + 8

    def test_code_starts_with_seven(self):
        c = Mad(Point)
        assert c.code.startswith("7")

    def test_schema_code_stable_across_instances(self):
        c1 = Mad(Point)
        c2 = Mad(Point)
        assert c1.code == c2.code


# =============================================================================
# Dict / map
# =============================================================================


class TestDict:
    def test_str_to_int_roundtrip(self):
        c = Mad(Dict[str, int])
        d = {"a": 1, "b": 2, "c": 3}
        assert roundtrip(c, d) == d

    def test_empty_dict_roundtrip(self):
        c = Mad(Dict[str, int])
        assert roundtrip(c, {}) == {}

    def test_none_encodes_as_zero_count(self):
        c = Mad(Dict[str, int])
        raw = c.encode(None)
        count = struct.unpack(">I", raw[:4])[0]
        assert count == 0

    def test_none_decodes_to_empty_dict(self):
        c = Mad(Dict[str, int])
        result = roundtrip(c, None)
        assert result == {}

    def test_count_prefix_is_correct(self):
        c = Mad(Dict[str, int])
        raw = c.encode({"x": 1, "y": 2})
        count = struct.unpack(">I", raw[:4])[0]
        assert count == 2

    def test_size_none_is_four(self):
        c = Mad(Dict[str, int])
        assert c.get_required_size(None) == 4

    def test_size_non_empty(self):
        c = Mad(Dict[str, int])
        # {"hi": 99}: 4 (count) + (4+2) (key "hi") + 8 (value int64) = 18
        assert c.get_required_size({"hi": 99}) == 18

    def test_int_to_str_roundtrip(self):
        c = Mad(Dict[int, str])
        d = {1: "one", 2: "two"}
        assert roundtrip(c, d) == d

    def test_code_starts_with_eight(self):
        c = Mad(Dict[str, int])
        assert c.code.startswith("8")

    def test_decode_truncated_raises(self):
        c = Mad(Dict[str, int])
        with pytest.raises(ValueError, match="[Bb]uffer"):
            c.decode(b"\x00\x00")  # too short for 4-byte count

    def test_large_dict_roundtrip(self):
        c = Mad(Dict[str, int])
        d = {str(i): i for i in range(500)}
        assert roundtrip(c, d) == d


# =============================================================================
# List (variable-length)
# =============================================================================


class TestList:
    def test_int_list_roundtrip(self):
        c = Mad(List[int])
        assert roundtrip(c, [1, 2, 3]) == [1, 2, 3]

    def test_empty_list_roundtrip(self):
        c = Mad(List[int])
        assert roundtrip(c, []) == []

    def test_str_list_roundtrip(self):
        c = Mad(List[str])
        assert roundtrip(c, ["hello", "world", ""]) == ["hello", "world", ""]

    def test_count_prefix_present(self):
        c = Mad(List[int])
        raw = c.encode([10, 20])
        count = struct.unpack(">I", raw[:4])[0]
        assert count == 2

    def test_size_empty_is_four(self):
        c = Mad(List[int])
        assert c.get_required_size([]) == 4

    def test_size_non_empty(self):
        c = Mad(List[int])
        # 4 (count) + 3 * 8 (int64) = 28
        assert c.get_required_size([1, 2, 3]) == 28

    def test_nested_list_of_dataclass(self):
        c = Mad(List[Point])  # type: ignore[type-arg]
        pts = [Point(1.0, 2.0), Point(-1.0, 0.5)]
        assert roundtrip(c, pts) == pts

    def test_large_list_roundtrip(self):
        c = Mad(List[int])
        lst = list(range(1000))
        assert roundtrip(c, lst) == lst

    def test_code_starts_with_six(self):
        c = Mad(List[int])
        assert c.code.startswith("6")


# =============================================================================
# FixedArray
# =============================================================================


@dataclass
class Vec3:
    coords: FixedArray(float, 3)


@dataclass
class ByteRow:
    data: FixedArray(Annotated[int, "uint8"], 4)


class TestFixedArray:
    def test_float_vec3_roundtrip(self):
        c = Mad(Vec3)
        v = Vec3([1.0, 2.0, 3.0])
        assert roundtrip(c, v) == v

    def test_uint8_row_roundtrip(self):
        c = Mad(ByteRow)
        v = ByteRow([0, 127, 128, 255])
        assert roundtrip(c, v) == v

    def test_size_is_element_size_times_length(self):
        c = Mad(Vec3)
        # 3 float64s = 24 bytes
        assert c.get_required_size(Vec3([0.0, 0.0, 0.0])) == 24

    def test_no_length_prefix(self):
        c = Mad(Vec3)
        raw = c.encode(Vec3([1.0, 2.0, 3.0]))
        # Raw should be exactly 24 bytes (3 × 8), no 4-byte count prefix
        assert len(raw) == 24

    def test_wrong_length_raises(self):
        c = Mad(Vec3)
        with pytest.raises(ValueError, match="length"):
            c.encode(Vec3([1.0, 2.0]))  # too short

    def test_code_starts_with_five(self):
        c = Mad(Vec3)
        assert c.code.startswith("7")  # outer struct "7", inner FixedArray "5..."


# =============================================================================
# Nested / complex structures
# =============================================================================


@dataclass
class Address:
    city: str
    zip_code: Annotated[int, "uint32"]


@dataclass
class Person:
    age: Annotated[int, "uint8"]
    address: Address
    name: str


@dataclass
class Inventory:
    counts: Dict[str, Annotated[int, "int32"]]
    tags: List[str]


class TestComplexStructures:
    def test_person_roundtrip(self):
        c = Mad(Person)
        p = Person(age=30, address=Address("Montreal", 12345), name="Alice")
        assert roundtrip(c, p) == p

    def test_inventory_roundtrip(self):
        c = Mad(Inventory)
        inv = Inventory(counts={"apples": 5, "bananas": 3}, tags=["fresh", "organic"])
        assert roundtrip(c, inv) == inv

    def test_person_size_is_sum_of_parts(self):
        c = Mad(Person)
        p = Person(age=30, address=Address("MTL", 99999), name="Bob")
        raw = c.encode(p)
        assert len(raw) == c.get_required_size(p)

    def test_encoded_length_matches_get_required_size(self):
        """encode() output length must always equal get_required_size()."""
        codecs_and_values = [
            (Mad(Point), Point(1.0, 2.0)),
            (Mad(str), "test string"),
            (Mad(Dict[str, int]), {"k": 1}),
            (Mad(List[str]), ["a", "bb", "ccc"]),
            (Mad(Person), Person(25, Address("NYC", 10001), "Carol")),
        ]
        for codec, value in codecs_and_values:
            raw = codec.encode(value)
            assert len(raw) == codec.get_required_size(value), (
                f"Mismatch for {value!r}: encoded {len(raw)} bytes "
                f"but get_required_size returned {codec.get_required_size(value)}"
            )


# =============================================================================
# Schema codes
# =============================================================================


class TestSchemaCodes:
    def test_bool_code(self):
        assert Mad(bool).code == "0"

    def test_int_default_code(self):
        assert Mad(int).code == "3"

    def test_float_default_code(self):
        assert Mad(float).code == "3"

    def test_str_code(self):
        assert Mad(str).code == "4"

    def test_list_int_code(self):
        assert Mad(List[int]).code == "63"

    def test_dict_str_int_code(self):
        assert Mad(Dict[str, int]).code == "843"

    def test_struct_code_starts_with_seven(self):
        assert Mad(Point).code.startswith("7")

    def test_fixed_array_code_starts_with_five(self):
        # Vec3 is a struct containing a FixedArray[float64], code = "7" + "5" + "3"
        assert "53" in Mad(Vec3).code

    def test_different_schemas_have_different_codes(self):
        codes = {
            Mad(bool).code,
            Mad(int).code,
            Mad(str).code,
            Mad(List[int]).code,
            Mad(Dict[str, int]).code,
        }
        assert len(codes) == 5  # all distinct


# =============================================================================
# Error handling
# =============================================================================


class TestErrors:
    def test_unsupported_type_raises_type_error(self):
        with pytest.raises(TypeError, match="[Uu]nsupported"):
            Mad(set)

    def test_non_dataclass_struct_strategy_raises(self):
        # A plain class (not dataclass) should raise TypeError
        class Plain:
            x: int

        with pytest.raises(TypeError):
            Mad(Plain)

    def test_decode_empty_bytes_raises_for_bool(self):
        c = Mad(bool)
        with pytest.raises(ValueError, match="[Bb]uffer"):
            c.decode(b"")

    def test_decode_empty_bytes_raises_for_int(self):
        c = Mad(int)
        with pytest.raises(ValueError, match="[Bb]uffer"):
            c.decode(b"")

    def test_decode_truncated_struct_raises(self):
        c = Mad(Point)
        raw = c.encode(Point(1.0, 2.0))
        with pytest.raises(ValueError, match="[Bb]uffer"):
            c.decode(raw[:4])  # only half of first float64

    def test_fixed_array_wrong_length_raises(self):
        c = Mad(Vec3)
        with pytest.raises(ValueError, match="length"):
            c.encode(Vec3([1.0, 2.0, 3.0, 4.0]))  # too long


# =============================================================================
# Regression: FixedArray inside a dataclass field
# (was broken due to _resolve_annotation filtering tuple metadata)
# =============================================================================


@dataclass
class Matrix2x2:
    row0: FixedArray(float, 2)
    row1: FixedArray(float, 2)


class TestFixedArrayInStruct:
    def test_matrix_roundtrip(self):
        c = Mad(Matrix2x2)
        m = Matrix2x2(row0=[1.0, 2.0], row1=[3.0, 4.0])
        assert roundtrip(c, m) == m

    def test_matrix_size(self):
        c = Mad(Matrix2x2)
        # 2 rows × 2 floats × 8 bytes = 32
        assert c.get_required_size(Matrix2x2([0.0, 0.0], [0.0, 0.0])) == 32

    def test_matrix_no_length_prefix_in_raw_bytes(self):
        c = Mad(Matrix2x2)
        raw = c.encode(Matrix2x2([1.0, 2.0], [3.0, 4.0]))
        # Exactly 4 float64s, no count prefixes
        assert len(raw) == 32


# =============================================================================
# Idempotency and determinism
# =============================================================================


class TestDeterminism:
    def test_same_value_same_bytes(self):
        c = Mad(Point)
        v = Point(1.23, 4.56)
        assert c.encode(v) == c.encode(v)

    def test_dict_encode_is_deterministic(self):
        # Python 3.7+ dicts preserve insertion order, so repeated encodes are stable
        c = Mad(Dict[str, int])
        d = {"a": 1, "b": 2}
        assert c.encode(d) == c.encode(d)

    def test_encode_decode_is_identity(self):
        c = Mad(Person)
        p = Person(20, Address("Paris", 75001), "Jean")
        assert roundtrip(c, roundtrip(c, p)) == p  # double roundtrip
