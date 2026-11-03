import unittest

import numpy as np

from fourier import FourierSeries, resample_closed_path


class SamplingTests(unittest.TestCase):
    def test_square_is_sampled_at_equal_arc_distances(self):
        samples = resample_closed_path([0j, 4, 4 + 4j, 4j], 8)
        np.testing.assert_allclose(samples, [0, 2, 4, 4 + 2j, 4 + 4j, 2 + 4j, 4j, 2j])

    def test_stationary_points_and_explicit_closure_do_not_change_sampling(self):
        plain = resample_closed_path([0j, 4, 4 + 4j, 4j], 32)
        repeated = resample_closed_path([0j, 0j, 2, 4, 4, 4 + 4j, 4j, 0j, 0j], 32)
        np.testing.assert_allclose(plain, repeated)

    def test_invalid_paths_fail_cleanly(self):
        for points in ([], [0j], [2j, 2j, 2j], [0j, complex(np.nan, 0)],
                       [0j, complex(0, np.inf)]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                resample_closed_path(points)
        for count in (0, 1, 2.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                resample_closed_path([0j, 1], count)


class FourierTests(unittest.TestCase):
    def setUp(self):
        self.series = FourierSeries.from_points([0j, 150, 120 + 60j, 40 + 90j], 128)

    def test_all_terms_reproduce_the_original_samples_in_order(self):
        phases = np.arange(128) / 128
        np.testing.assert_allclose(self.series.reconstruct(phases), self.series.samples, atol=1e-10)

    def test_signed_frequencies_preserve_clockwise_and_counterclockwise_motion(self):
        phases = np.arange(64) / 64
        for frequency in (-3, 2):
            with self.subTest(frequency=frequency):
                points = 50 + 30j + 20 * np.exp(2j * np.pi * frequency * phases)
                series = FourierSeries.from_points(points, 64)
                self.assertEqual(series.frequencies[0], frequency)
                np.testing.assert_allclose(series.reconstruct(phases), points, atol=1e-10)
                np.testing.assert_allclose(series.chain(0.25)[-1],
                                           50 + 30j + 20 * np.exp(2j * np.pi * frequency * 0.25),
                                           atol=1e-10)

    def test_chain_endpoint_matches_reconstruction(self):
        for terms in (0, 1, 16, 128):
            with self.subTest(terms=terms):
                actual = self.series.chain(0.37, terms)
                self.assertEqual(len(actual), terms + 1)
                np.testing.assert_allclose(actual[-1], self.series.reconstruct([0.37], terms)[0])

    def test_more_terms_reduce_error_and_parseval_matches_sample_error(self):
        phases = np.arange(128) / 128
        errors = []
        for terms in (0, 1, 4, 16, 64, 128):
            reconstructed = self.series.reconstruct(phases, terms)
            measured = np.sqrt(np.mean(np.abs(reconstructed - self.series.samples) ** 2))
            self.assertAlmostEqual(self.series.rms_error(terms), measured, places=9)
            errors.append(measured)
        self.assertTrue(all(a >= b - 1e-10 for a, b in zip(errors, errors[1:])))
        self.assertLess(errors[-1], 1e-10)

    def test_translation_only_moves_the_origin(self):
        points = [0j, 150, 120 + 60j, 40 + 90j]
        shifted = FourierSeries.from_points([point + 30 - 7j for point in points], 128)
        self.assertAlmostEqual(shifted.origin - self.series.origin, 30 - 7j)
        # Equal-amplitude ties may be sorted differently; compare reconstructed curves.
        np.testing.assert_allclose(shifted.reconstruct([0.1, 0.4]),
                                   self.series.reconstruct([0.1, 0.4]) + 30 - 7j, atol=1e-10)

    def test_phase_wraps_after_one_cycle(self):
        np.testing.assert_allclose(self.series.chain(0.2), self.series.chain(1.2), atol=1e-10)

    def test_invalid_term_counts_are_rejected(self):
        for terms in (-1, 129, 3.5):
            with self.subTest(terms=terms), self.assertRaises(ValueError):
                self.series.chain(0, terms)


if __name__ == '__main__':
    unittest.main()
