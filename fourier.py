"""Arc-length sampling and complex Fourier reconstruction, independent of Pygame."""

from dataclasses import dataclass

import numpy as np


def resample_closed_path(points, count=256):
    """Sample a closed polyline at equal distances, ignoring stationary input."""
    if not isinstance(count, int) or count < 2:
        raise ValueError('Sample count must be an integer of at least 2')
    vertices = np.asarray(list(points), dtype=np.complex128)
    if vertices.ndim != 1 or len(vertices) < 2 or not np.isfinite(vertices).all():
        raise ValueError('Draw at least two distinct, finite points')
    vertices = np.append(vertices, vertices[0])
    lengths = np.abs(np.diff(vertices))
    moving = lengths > 1e-9
    if not moving.any():
        raise ValueError('Draw a line or curve, rather than a single point')
    starts, ends = vertices[:-1][moving], vertices[1:][moving]
    boundaries = np.concatenate(([0.0], np.cumsum(lengths[moving])))
    vertices = np.concatenate((starts[:1], ends))
    distances = np.linspace(0, boundaries[-1], count, endpoint=False)
    return (np.interp(distances, boundaries, vertices.real)
            + 1j * np.interp(distances, boundaries, vertices.imag))


@dataclass
class FourierSeries:
    origin: complex
    samples: np.ndarray
    frequencies: np.ndarray
    coefficients: np.ndarray

    @classmethod
    def from_points(cls, points, count=256):
        samples = resample_closed_path(points, count)
        origin = complex(samples.mean())
        coefficients = np.fft.fft(samples - origin) / count
        frequencies = np.rint(np.fft.fftfreq(count) * count).astype(int)
        order = np.argsort(-np.abs(coefficients), kind='stable')
        return cls(origin, samples, frequencies[order], coefficients[order])

    def _terms(self, terms):
        if terms is None:
            return len(self.coefficients)
        if not isinstance(terms, int) or not 0 <= terms <= len(self.coefficients):
            raise ValueError('Terms must be between 0 and the sample count')
        return terms

    def chain(self, phase, terms=None):
        """Circle centers followed by the final endpoint; phase is in cycles."""
        count = self._terms(terms)
        vectors = self.coefficients[:count] * np.exp(
            2j * np.pi * self.frequencies[:count] * phase)
        return self.origin + np.concatenate(([0j], np.cumsum(vectors)))

    def reconstruct(self, phases, terms=None):
        count = self._terms(terms)
        angles = 2j * np.pi * np.outer(phases, self.frequencies[:count])
        return self.origin + np.exp(angles) @ self.coefficients[:count]

    def rms_error(self, terms=None):
        """RMS distance from the sampled curve, in pixels (Parseval's identity)."""
        count = self._terms(terms)
        return float(np.sqrt(np.sum(np.abs(self.coefficients[count:]) ** 2)))
