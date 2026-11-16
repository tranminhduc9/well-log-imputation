"""GeoLink NULL decoding must not discard valid negative or positive logs."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import lasio
import numpy as np

from src.preprocessing.read_data import read_data, read_data_directory


class LASNullTests(unittest.TestCase):
    def test_single_and_directory_readers_decode_both_null_codes(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "example.las"
            las = lasio.LASFile()
            las.well.WELL.value = "example"
            las.well.NULL.value = -999.25
            las.append_curve("DEPT", [-999., 1001., 1002., 1003.], unit="m")
            las.append_curve("GR", [10., -999., -999.25, 20.], unit="gAPI")
            las.append_curve("NPHI", [-.05, 0., .5, 112.014], unit="m3/m3")
            las.write(str(path), version=2, wrap=False)
            for frame in [read_data(path), read_data_directory(temporary)]:
                with self.subTest(columns=list(frame)):
                    np.testing.assert_allclose(frame.GR, [10., np.nan, np.nan, 20.], equal_nan=True)
                    np.testing.assert_allclose(frame.NPHI, [-.05, 0., .5, 112.014])
                    self.assertEqual(frame.DEPT.iloc[0], -999.)


if __name__ == "__main__":
    unittest.main()
