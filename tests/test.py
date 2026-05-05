import unittest
from dataclasses import dataclass
from mad.core import Mad, MadType

@dataclass
class Stats:
    level: MadType.uint16
    xp: MadType.uint64

@dataclass
class Player:
    name: MadType.string
    stats: Stats
    inventory: dict[MadType.string, MadType.uint32]
    position: tuple[MadType.float32, 3]
    is_active: MadType.bool

class TestMadSerialization(unittest.TestCase):

    def test_primitives(self):
        """Test all basic MadType primitives."""
        test_cases = [
            (MadType.bool, True),
            (MadType.int8, -127),
            (MadType.uint8, 255),
            (MadType.int16, -32767),
            (MadType.uint16, 65535),
            (MadType.int32, -2147483648),
            (MadType.uint32, 4294967295),
            (MadType.float32, 3.14159),
            (MadType.int64, -9223372036854775808),
            (MadType.uint64, 18446744073709551615),
            (MadType.float64, 2.718281828459),
            (MadType.string, "Hello, Mad Serialization! 🚀"),
        ]

        for mtype, value in test_cases:
            with self.subTest(mtype=mtype):
                m = Mad(mtype)
                encoded = m.encode(value)
                decoded = m.decode(encoded)
                if isinstance(value, float):
                    self.assertAlmostEqual(decoded, value, places=5)
                else:
                    self.assertEqual(decoded, value)

    def test_fixed_array(self):
        """Test tuple[T, N] representation of fixed arrays."""
        m = Mad(tuple[MadType.int32, 4])
        data = [10, -20, 30, -40]
        encoded = m.encode(data)
        self.assertEqual(len(encoded), 16)  # 4 * 4 bytes
        self.assertEqual(m.decode(encoded), data)

        # Test validation
        with self.assertRaises(ValueError):
            m.encode([1, 2, 3]) # Wrong length

    def test_dictionary_map(self):
        """Test dict[K, V] serialization."""
        m = Mad(dict[MadType.string, MadType.int32])
        data = {"health": 100, "mana": 50, "stamina": 75}
        encoded = m.encode(data)
        decoded = m.decode(encoded)
        self.assertEqual(decoded, data)

    def test_nested_dataclass(self):
        """Test complex nested structs with mixed types."""
        m = Mad(Player)
        
        player_instance = Player(
            name="MadMax",
            stats=Stats(level=50, xp=1200000),
            inventory={"Sword": 1, "Potion": 15, "Gold": 5000},
            position=(123.45, 67.89, 0.0),
            is_active=True
        )

        encoded = m.encode(player_instance)
        decoded = m.decode(encoded)

        # Check fields
        self.assertEqual(decoded.name, player_instance.name)
        self.assertEqual(decoded.stats.level, player_instance.stats.level)
        self.assertEqual(decoded.stats.xp, player_instance.stats.xp)
        self.assertEqual(decoded.inventory, player_instance.inventory)
        self.assertEqual(decoded.is_active, player_instance.is_active)
        
        # Check tuple elements
        for i in range(3):
            self.assertAlmostEqual(decoded.position[i], player_instance.position[i], places=3)

    def test_encode_into_buffer(self):
        """Test serialization into a pre-allocated bytearray."""
        m = Mad(MadType.uint32)
        buf = bytearray(4)
        m.encode_into(123456, buf)
        self.assertEqual(m.decode(bytes(buf)), 123456)

        # Test buffer too small
        small_buf = bytearray(2)
        with self.assertRaises(ValueError):
            m.encode_into(123456, small_buf)


if __name__ == "__main__":
    unittest.main()
