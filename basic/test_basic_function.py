import unittest
from basic_function import increment

class TestBasicFunction(unittest.TestCase):
    def test_increment(self):
        self.assertEqual(increment(3), 4)
        self.assertEqual(increment(-1), 0)
        self.assertEqual(increment(0), 1)

if __name__ == '__main__':
    unittest.main()
