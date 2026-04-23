"""
mad.py — Python port of the Go `mad` binary serialization library.

Supported types:
  Primitives : bool, int8/16/32/64, uint8/16/32/64, float32, float64
               (use typing.Annotated to carry width/signedness hints)
  str        : 4-byte length prefix + UTF-8 bytes
  dataclass  : fields sorted alphabetically, each field serialized recursively
  list / tuple with fixed length: annotated as FixedArray[T, N]
  dict       : 4-byte count prefix + alternating key/value pairs

Usage
-----
from dataclasses import dataclass
from mad import Mad

@dataclass
class Point:
    x: float   # treated as float64
    y: float

codec = Mad(Point)
buf = codec.encode(Point(1.0, 2.0))
decoded = codec.decode(buf)
"""

import dataclasses
import struct
import typing
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    Tuple,
    Type,
    TypeVar,
    get_type_hints,
)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Type annotation helpers
# ---------------------------------------------------------------------------

# Use typing.Annotated to carry extra width/signedness info, e.g.:
#   Annotated[int, "int8"]   Annotated[int, "uint32"]   Annotated[float, "float32"]
# If no annotation is given, int → int64, float → float64.

_INT_FORMATS: Dict[str, Tuple[str, int, str]] = {
    # name        : (struct_fmt, byte_size, code)
    "int8": (">b", 1, "0"),
    "uint8": (">B", 1, "0"),
    "bool": (">?", 1, "0"),
    "int16": (">h", 2, "1"),
    "uint16": (">H", 2, "1"),
    "int32": (">i", 4, "2"),
    "uint32": (">I", 4, "2"),
    "float32": (">f", 4, "2"),
    "int64": (">q", 8, "3"),
    "uint64": (">Q", 8, "3"),
    "float64": (">d", 8, "3"),
}


# ---------------------------------------------------------------------------
# Strategy return type
# ---------------------------------------------------------------------------
# Each strategy returns (encode_fn, decode_fn, size_fn, code_str)
# encode_fn : (value) -> bytes
# decode_fn : (memoryview) -> (value, bytes_consumed)
# size_fn   : (value) -> int

Strategy = Tuple[
    Callable[[Any], bytes],
    Callable[[memoryview], Tuple[Any, int]],
    Callable[[Any], int],
    str,
]


# ---------------------------------------------------------------------------
# Primitive strategies
# ---------------------------------------------------------------------------


def _fixed_strategy(fmt: str, size: int, code: str) -> Strategy:
    _pack = struct.Struct(fmt)

    def encode(value: Any) -> bytes:
        return _pack.pack(value)

    def decode(buf: memoryview) -> Tuple[Any, int]:
        if len(buf) < size:
            raise ValueError(f"Buffer too small: need {size}, got {len(buf)}")
        return _pack.unpack_from(buf)[0], size

    def sizefn(_value: Any) -> int:
        return size

    return encode, decode, sizefn, code


def _string_strategy() -> Strategy:
    _len_struct = struct.Struct(">I")

    def encode(value: str) -> bytes:
        encoded = value.encode("utf-8")
        return _len_struct.pack(len(encoded)) + encoded

    def decode(buf: memoryview) -> Tuple[str, int]:
        if len(buf) < 4:
            raise ValueError("Buffer too small for string length prefix")
        (n,) = _len_struct.unpack_from(buf)
        if len(buf) < 4 + n:
            raise ValueError("Buffer too small for string data")
        return bytes(buf[4 : 4 + n]).decode("utf-8"), 4 + n

    def sizefn(value: str) -> int:
        return 4 + len(value.encode("utf-8"))

    return encode, decode, sizefn, "4"


# ---------------------------------------------------------------------------
# Composite strategies
# ---------------------------------------------------------------------------


def _array_strategy(element_strategy: Strategy, length: int) -> Strategy:
    enc_elem, dec_elem, size_elem, elem_code = element_strategy

    def encode(value: Any) -> bytes:
        if len(value) != length:
            raise ValueError(f"Expected array of length {length}, got {len(value)}")
        parts = [enc_elem(v) for v in value]
        return b"".join(parts)

    def decode(buf: memoryview) -> Tuple[Any, int]:
        result = []
        offset = 0
        for _ in range(length):
            val, consumed = dec_elem(buf[offset:])
            result.append(val)
            offset += consumed
        return result, offset

    def sizefn(value: Any) -> int:
        return sum(size_elem(v) for v in value)

    return encode, decode, sizefn, "5" + elem_code


def _struct_strategy(typ: type) -> Strategy:
    if not dataclasses.is_dataclass(typ):
        raise TypeError(f"Struct strategy requires a dataclass, got {typ}")

    hints = get_type_hints(typ, include_extras=True)
    fields = dataclasses.fields(typ)

    # Sort fields alphabetically (mirrors Go struct sort)
    sorted_fields = sorted(fields, key=lambda f: f.name)

    field_strategies = []
    code = "7"
    for field in sorted_fields:
        hint = hints.get(field.name, field.type)
        strat = _generate_strategy(hint)
        field_strategies.append((field.name, strat))
        code += strat[3]

    def encode(value: Any) -> bytes:
        parts = []
        for name, (enc, _dec, _size, _code) in field_strategies:
            parts.append(enc(getattr(value, name)))
        return b"".join(parts)

    def decode(buf: memoryview) -> Tuple[Any, int]:
        offset = 0
        kwargs = {}
        for name, (_enc, dec, _size, _code) in field_strategies:
            val, consumed = dec(buf[offset:])
            kwargs[name] = val
            offset += consumed
        return typ(**kwargs), offset

    def sizefn(value: Any) -> int:
        total = 0
        for name, (_enc, _dec, size, _code) in field_strategies:
            total += size(getattr(value, name))
        return total

    return encode, decode, sizefn, code


def _map_strategy(key_strategy: Strategy, val_strategy: Strategy) -> Strategy:
    enc_key, dec_key, size_key, key_code = key_strategy
    enc_val, dec_val, size_val, val_code = val_strategy
    _count_struct = struct.Struct(">I")

    def encode(value: Optional[dict]) -> bytes:
        if value is None:
            return _count_struct.pack(0)
        parts = [_count_struct.pack(len(value))]
        for k, v in value.items():
            parts.append(enc_key(k))
            parts.append(enc_val(v))
        return b"".join(parts)

    def decode(buf: memoryview) -> Tuple[Optional[dict], int]:
        if len(buf) < 4:
            raise ValueError("Buffer too small for map count")
        (count,) = _count_struct.unpack_from(buf)
        offset = 4
        result = {}
        for _ in range(count):
            k, kc = dec_key(buf[offset:])
            offset += kc
            v, vc = dec_val(buf[offset:])
            offset += vc
            result[k] = v
        return result, offset

    def sizefn(value: Optional[dict]) -> int:
        if value is None:
            return 4
        total = 4
        for k, v in value.items():
            total += size_key(k) + size_val(v)
        return total

    return encode, decode, sizefn, "8" + key_code + val_code


# ---------------------------------------------------------------------------
# Type resolver
# ---------------------------------------------------------------------------


def _resolve_annotation(hint: Any) -> Tuple[type, Any]:
    """Return (base_type, metadata_or_None).

    metadata is the first element of __metadata__ (a string for primitive-width
    hints, a tuple for FixedArray hints, or None for plain types).
    """
    if getattr(hint, "__metadata__", None) is not None:
        # typing.Annotated
        base = hint.__args__[0]
        meta = hint.__metadata__[0] if hint.__metadata__ else None
        return base, meta
    return hint, None


def _generate_strategy(hint: Any) -> Strategy:
    base, annotation = _resolve_annotation(hint)
    origin = getattr(base, "__origin__", None)

    # --- bool must come before int (bool is a subclass of int in Python) ---
    if base is bool or annotation == "bool":
        return _fixed_strategy(">?", 1, "0")

    # --- annotated numeric primitives ---
    if annotation and annotation in _INT_FORMATS:
        fmt, size, code = _INT_FORMATS[annotation]
        return _fixed_strategy(fmt, size, code)

    # --- plain Python types with defaults ---
    if base is int:
        return _fixed_strategy(">q", 8, "3")  # default: int64
    if base is float:
        return _fixed_strategy(">d", 8, "3")  # default: float64
    if base is str:
        return _string_strategy()

    # --- dict ---
    if origin is dict:
        args = base.__args__
        key_strat = _generate_strategy(args[0])
        val_strat = _generate_strategy(args[1])
        return _map_strategy(key_strat, val_strat)

    # --- list/tuple (variable length) — encoded with 4-byte length prefix ---
    if origin in (list, tuple):
        args = base.__args__
        elem_strat = _generate_strategy(args[0])
        enc_elem, dec_elem, size_elem, elem_code = elem_strat
        _count_struct = struct.Struct(">I")

        def encode_list(value: Any) -> bytes:
            parts = [_count_struct.pack(len(value))]
            for v in value:
                parts.append(enc_elem(v))
            return b"".join(parts)

        def decode_list(buf: memoryview) -> Tuple[Any, int]:
            if len(buf) < 4:
                raise ValueError("Buffer too small for list length prefix")
            (n,) = _count_struct.unpack_from(buf)
            offset = 4
            result = []
            for _ in range(n):
                val, consumed = dec_elem(buf[offset:])
                result.append(val)
                offset += consumed
            return result, offset

        def size_list(value: Any) -> int:
            return 4 + sum(size_elem(v) for v in value)

        return encode_list, decode_list, size_list, "6" + elem_code

    # --- dataclass → struct strategy ---
    if dataclasses.is_dataclass(base):
        return _struct_strategy(base)

    raise TypeError(
        f"Unsupported type: {hint!r}. "
        "Supported: bool, int, float, str, dict, list, tuple, dataclass, "
        "or Annotated[int, 'int8'/'uint16'/etc.]"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class Mad(Generic[T]):
    """
    Binary codec for a given type T.

    Parameters
    ----------
    typ : type
        The Python type (dataclass, primitive, dict, list, …) to encode/decode.

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> from mad import Mad
    >>> @dataclass
    ... class Point:
    ...     x: float
    ...     y: float
    >>> codec = Mad(Point)
    >>> buf = codec.encode(Point(1.0, 2.0))
    >>> codec.decode(buf)
    Point(x=1.0, y=2.0)
    """

    def __init__(self, typ: Type[T]) -> None:
        strat = _generate_strategy(typ)
        self._encode_fn, self._decode_fn, self._size_fn, self._code = strat
        self._typ = typ

    @property
    def code(self) -> str:
        """Schema fingerprint string (mirrors Go's Mad.Code())."""
        return self._code

    def get_required_size(self, value: T) -> int:
        """Return the number of bytes needed to encode *value*."""
        return self._size_fn(value)

    def encode(self, value: T) -> bytes:
        """Serialize *value* to bytes."""
        return self._encode_fn(value)

    def decode(self, data: bytes) -> T:
        """Deserialize *data* and return the reconstructed value."""
        mv = memoryview(data)
        result, _ = self._decode_fn(mv)
        return result


# ---------------------------------------------------------------------------
# Convenience: FixedArray annotation (mirrors Go [N]T arrays)
# ---------------------------------------------------------------------------


class _FixedArrayMeta:
    """Marker used by _generate_strategy to detect fixed-length arrays."""


def FixedArray(element_type: type, length: int) -> Any:
    """
    Return an Annotated type hint for a fixed-length array.

    Example
    -------
    @dataclass
    class Vec3:
        coords: FixedArray(float, 3)
    """
    import typing

    return typing.Annotated[list, ("fixed_array", element_type, length)]


# Hook fixed-array annotation into the strategy resolver
_orig_generate = _generate_strategy


def _generate_strategy(hint: Any) -> Strategy:  # type: ignore[misc]
    base, annotation = _resolve_annotation(hint)
    if (
        isinstance(annotation, tuple)
        and len(annotation) == 3
        and annotation[0] == "fixed_array"
    ):
        _, elem_type, length = annotation
        elem_strat = _generate_strategy(elem_type)
        return _array_strategy(elem_strat, length)
    return _orig_generate(hint)
