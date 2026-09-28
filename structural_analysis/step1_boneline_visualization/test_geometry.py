import unittest
from build import validate_line


class GeometryTests(unittest.TestCase):
    def test_valid_subpixel(self):
        self.assertEqual(validate_line([[0, 0], [9.9, 9.9]], 10, 10), [])

    def test_invalid_geometry(self):
        for line in (None, [], [[1, 1]], [[1, 1], [1, 1]], [[0, 0], [10, 1]],
                     [[0, 0], [-1, 2]], [[0, 0], [1, 2, 3]], [[0, 0], [True, 2]],
                     [[0, 0], [float('nan'), 2]], [[0, 0], [float('inf'), 2]],
                     [[0, 0], ['1', 2]], [[0, 0], [1, 1], None, [2, 2]]):
            with self.subTest(line=line):
                self.assertTrue(validate_line(line, 10, 10))


if __name__ == '__main__':
    unittest.main()
