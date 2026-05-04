"""
mad.py — Binary serialization library.

Every type is expressed via MadType enum members or composites built from them.

Wire format (big-endian):
  bool / int8 / uint8       → 1 byte
  int16 / uint16            → 2 bytes
  int32 / uint32 / float32  → 4 bytes
  int64 / uint64 / float64  → 8 bytes
  string                    → 4-byte length prefix + UTF-8 bytes
  dataclass                 → fields serialized in alphabetical order by name
  tuple[T, N]               → N elements serialized in order
  dict[K, V]                → 4-byte count + (key, value) pairs

Usage
-----
Primitive:
    m = Mad(MadType.uint32)
    buf = m.encode(42)
    m.decode(buf)           # → 42

Dataclass:
    @dataclass
    class Player:
        name:  MadType.string
        score: MadType.uint32
        ratio: MadType.float32

    m = Mad(Player)
    m.decode(m.encode(Player(name="Alice", score=9999, ratio=0.95)))

Nested dataclass:
    @dataclass
    class World:
        label:  MadType.string
        player: Player

    m = Mad(World)

Fixed array:
    @dataclass
    class Transform:
        position: tuple[MadType.float32, 3]
        rotation: tuple[MadType.float32, 4]

Map:
    @dataclass
    class Inventory:
        items: dict[MadType.string, MadType.uint32]
"""

import struct
import dataclasses
from enum import Enum
from typing import Any, Callable, Tuple


# ---------------------------------------------------------------------------
# Internal aliases
# ---------------------------------------------------------------------------
EncFn  = Callable[[Any], bytes]
DecFn  = Callable[[Any], Tuple[Any, Any]]
SizeFn = Callable[[Any], int]

_CODES = {
    "byte":    "0",
    "two":     "1",
    "four":    "2",
    "eight":   "3",
    "string":  "4",
    "array":   "5",
    "struct":  "7",
    "map":     "8",
}


# ---------------------------------------------------------------------------
# MadType enum
# ---------------------------------------------------------------------------

class MadType(Enum):
    """All supported primitive wire types."""
    bool    = "bool"
    int8    = "int8"
    uint8   = "uint8"
    int16   = "int16"
    uint16  = "uint16"
    int32   = "int32"
    uint32  = "uint32"
    float32 = "float32"
    int64   = "int64"
    uint64  = "uint64"
    float64 = "float64"
    string  = "string"


# ---------------------------------------------------------------------------
# Primitive strategies
# ---------------------------------------------------------------------------

def _byte_strat(signed: bool = False):
    fmt = ">b" if signed else ">B"
    def enc(v): return struct.pack(fmt, int(v))
    def dec(buf):
        if len(buf) < 1: raise ValueError("buffer too small")
        (val,) = struct.unpack_from(fmt, buf)
        return val, buf[1:]
    def size(_): return 1
    return enc, dec, size, _CODES["byte"]

def _two_byte_strat(signed: bool = False):
    fmt = ">h" if signed else ">H"
    def enc(v): return struct.pack(fmt, v)
    def dec(buf):
        if len(buf) < 2: raise ValueError("buffer too small")
        (val,) = struct.unpack_from(fmt, buf)
        return val, buf[2:]
    def size(_): return 2
    return enc, dec, size, _CODES["two"]

def _four_byte_strat(is_float: bool = False, signed: bool = False):
    fmt = ">f" if is_float else (">i" if signed else ">I")
    def enc(v): return struct.pack(fmt, v)
    def dec(buf):
        if len(buf) < 4: raise ValueError("buffer too small")
        (val,) = struct.unpack_from(fmt, buf)
        return val, buf[4:]
    def size(_): return 4
    return enc, dec, size, _CODES["four"]

def _eight_byte_strat(is_float: bool = False, signed: bool = False):
    fmt = ">d" if is_float else (">q" if signed else ">Q")
    def enc(v): return struct.pack(fmt, v)
    def dec(buf):
        if len(buf) < 8: raise ValueError("buffer too small")
        (val,) = struct.unpack_from(fmt, buf)
        return val, buf[8:]
    def size(_): return 8
    return enc, dec, size, _CODES["eight"]

def _string_strat():
    def enc(v: str) -> bytes:
        b = v.encode("utf-8")
        return struct.pack(">I", len(b)) + b
    def dec(buf):
        if len(buf) < 4: raise ValueError("buffer too small")
        (n,) = struct.unpack_from(">I", buf)
        buf = buf[4:]
        if len(buf) < n: raise ValueError("buffer too small")
        return bytes(buf[:n]).decode("utf-8"), buf[n:]
    def size(v: str): return len(v.encode("utf-8")) + 4
    return enc, dec, size, _CODES["string"]


# ---------------------------------------------------------------------------
# MadType → strategy
# ---------------------------------------------------------------------------

_MADTYPE_STRATEGY: dict[MadType, Callable] = {
    MadType.bool:    lambda: _byte_strat(signed=False),
    MadType.int8:    lambda: _byte_strat(signed=True),
    MadType.uint8:   lambda: _byte_strat(signed=False),
    MadType.int16:   lambda: _two_byte_strat(signed=True),
    MadType.uint16:  lambda: _two_byte_strat(signed=False),
    MadType.int32:   lambda: _four_byte_strat(signed=True),
    MadType.uint32:  lambda: _four_byte_strat(signed=False),
    MadType.float32: lambda: _four_byte_strat(is_float=True),
    MadType.int64:   lambda: _eight_byte_strat(signed=True),
    MadType.uint64:  lambda: _eight_byte_strat(signed=False),
    MadType.float64: lambda: _eight_byte_strat(is_float=True),
    MadType.string:  lambda: _string_strat(),
}


# ---------------------------------------------------------------------------
# Composite strategies
# ---------------------------------------------------------------------------

def _array_strat(element_type: Any, arr_len: int):
    enc_e, dec_e, size_e, ecode = _generate_funcs(element_type)
    def enc(value):
        if len(value) != arr_len:
            raise ValueError(f"expected {arr_len} elements, got {len(value)}")
        return b"".join(enc_e(item) for item in value)
    def dec(buf):
        result = []
        for _ in range(arr_len):
            item, buf = dec_e(buf)
            result.append(item)
        return result, buf
    def size(value): return sum(size_e(item) for item in value)
    return enc, dec, size, _CODES["array"] + ecode


def _map_strat(key_type: Any, val_type: Any):
    enc_k, dec_k, size_k, kcode = _generate_funcs(key_type)
    enc_v, dec_v, size_v, vcode = _generate_funcs(val_type)
    def enc(value: dict) -> bytes:
        if value is None:
            return struct.pack(">I", 0)
        parts = [struct.pack(">I", len(value))]
        for k, v in value.items():
            parts.append(enc_k(k))
            parts.append(enc_v(v))
        return b"".join(parts)
    def dec(buf):
        if len(buf) < 4: raise ValueError("buffer too small")
        (count,) = struct.unpack_from(">I", buf)
        buf = buf[4:]
        result = {}
        for _ in range(count):
            k, buf = dec_k(buf)
            v, buf = dec_v(buf)
            result[k] = v
        return result, buf
    def size(value: dict) -> int:
        if value is None: return 4
        return 4 + sum(size_k(k) + size_v(v) for k, v in value.items())
    return enc, dec, size, _CODES["map"] + kcode + vcode


def _struct_strat(cls: type):
    raw_hints = cls.__annotations__
    sorted_fields = sorted(raw_hints.items(), key=lambda x: x[0])

    field_funcs = []
    code = _CODES["struct"]
    for name, ftype in sorted_fields:
        enc_f, dec_f, size_f, fcode = _generate_funcs(ftype)
        field_funcs.append((name, enc_f, dec_f, size_f))
        code += fcode

    def enc(value) -> bytes:
        return b"".join(enc_f(getattr(value, name)) for name, enc_f, _, _ in field_funcs)
    def dec(buf):
        kwargs = {}
        for name, _, dec_f, _ in field_funcs:
            val, buf = dec_f(buf)
            kwargs[name] = val
        return cls(**kwargs), buf
    def size(value) -> int:
        return sum(size_f(getattr(value, name)) for name, _, _, size_f in field_funcs)

    return enc, dec, size, code


# ---------------------------------------------------------------------------
# Central dispatcher
# ---------------------------------------------------------------------------

def _generate_funcs(typ: Any) -> Tuple[EncFn, DecFn, SizeFn, str]:
    if typ is None:
        raise TypeError("type cannot be None")

    # MadType enum member
    if isinstance(typ, MadType):
        return _MADTYPE_STRATEGY[typ]()

    # Generic aliases: dict[K, V] and tuple[T, N]
    origin = getattr(typ, "__origin__", None)
    args   = getattr(typ, "__args__", ())

    if origin is dict:
        return _map_strat(args[0], args[1])

    if origin is tuple:
        if len(args) == 2 and isinstance(args[1], int):
            return _array_strat(args[0], args[1])
        if args and all(a == args[0] for a in args):
            return _array_strat(args[0], len(args))
        raise TypeError(
            f"Use tuple[MadType.X, LENGTH] for fixed arrays, got {typ}."
        )

    if origin is list:
        raise TypeError("list is not supported. Use tuple[MadType.X, N] for fixed arrays.")

    # dataclass
    if dataclasses.is_dataclass(typ) and isinstance(typ, type):
        return _struct_strat(typ)

    raise TypeError(
        f"Unsupported type: {typ!r}\n"
        "Supported: MadType members, dataclasses, "
        "dict[MadType.X, MadType.Y], tuple[MadType.X, N]."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Mad:
    """
    Binary serializer / deserializer.

    Pass a MadType member for a primitive, or a dataclass for a struct.
    All field annotations must be MadType members, nested dataclasses,
    dict[...], or tuple[...].

    Parameters
    ----------
    typ : MadType | type
        The type to serialize/deserialize.

    Examples
    --------
    >>> m = Mad(MadType.uint32)
    >>> m.decode(m.encode(42))
    42

    >>> @dataclass
    ... class Player:
    ...     name:  MadType.string
    ...     score: MadType.uint32
    >>> m = Mad(Player)
    >>> m.decode(m.encode(Player(name="Alice", score=99)))
    Player(name='Alice', score=99)
    """

    def __init__(self, typ: "MadType | type"):
        enc, dec, size, code = _generate_funcs(typ)
        self._enc  = enc
        self._dec  = dec
        self._size = size
        self._code = code

    @property
    def code(self) -> str:
        """Schema fingerprint — changes when the layout changes."""
        return self._code

    def get_required_size(self, value: Any) -> int:
        """Number of bytes needed to encode *value*."""
        return self._size(value)

    def encode(self, value: Any) -> bytes:
        """Serialize *value* to bytes."""
        return self._enc(value)

    def encode_into(self, value: Any, output: bytearray) -> None:
        """Serialize *value* into a pre-allocated bytearray."""
        needed = self.get_required_size(value)
        if len(output) < needed:
            raise ValueError(f"buffer too small: need {needed}, got {len(output)}")
        output[:needed] = self._enc(value)

    def decode(self, data: bytes) -> Any:
        """Deserialize bytes back to the original value."""
        value, _ = self._dec(memoryview(data))
        return value
