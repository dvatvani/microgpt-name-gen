import unittest
from microgpt_name_gen.example import hello


class Testmicrogpt_name_gen(unittest.TestCase):
    def test_hello(self):
        assert hello("Giulio") == "Hello Giulio!"


if __name__ == "__main__":
    unittest.main()
