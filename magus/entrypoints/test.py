import os
import unittest
import magus
from BeautifulReport import BeautifulReport


def test(*args, totest="*", **kwargs):
    tests_path = os.path.join(magus.__path__[0], 'tests')
    discover = unittest.defaultTestLoader.discover(tests_path, pattern='test_{}.py'.format(totest))
    runner = BeautifulReport(discover)
    runner.report(
        description="MagusTest",
        filename="MagusTestReport"
    )
