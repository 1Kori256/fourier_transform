"""Draw a curve and reconstruct it with rotating Fourier epicycles."""

import argparse
import os
from pathlib import Path

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

import numpy as np
import pygame
import pygame.gfxdraw

from fourier import FourierSeries

SIZE = (1180, 780)
SAMPLES = 256
TRACE_SAMPLES = 1024
BACKGROUND = (12, 18, 28)
PANEL = (20, 29, 42)
BORDER = (39, 52, 68)
TEXT = (227, 236, 244)
MUTED = (132, 153, 175)
CYAN = (87, 226, 208)
ORANGE = (255, 184, 107)


def xy(point):
    return (round(point.real), round(point.imag))


class Visualizer:
    def __init__(self, surface):
        self.surface = surface
        self.canvas = pygame.Rect(24, 96, 816, 608)
        self.panel = pygame.Rect(860, 96, 296, 608)
        self.font = pygame.font.Font(None, 23)
        self.small = pygame.font.Font(None, 19)
        self.title = pygame.font.Font(None, 42)
        self.number = pygame.font.Font(None, 62)
        self.slider = pygame.Rect(884, 257, 248, 16)
        self.series = None
        self.points = []
        self.trace = []
        self.terms = 32
        self.phase = 0.0
        self.speed = 1.0
        self.paused = False
        self.show_circles = True
        self.show_original = True
        self.drawing = False
        self.dragging_slider = False
        self.demo_index = -1
        self.message = 'Draw anywhere on the canvas to replace the demo.'
        self.load_demo()

    def text(self, value, position, color=TEXT, font=None):
        self.surface.blit((font or self.font).render(value, True, color), position)

    def load_points(self, points):
        try:
            series = FourierSeries.from_points(points, SAMPLES)
        except ValueError as error:
            self.message = str(error)
            return False
        self.series = series
        self.phase = 0.0
        self.update_trace()
        self.message = 'Drag to draw another curve. Open strokes are closed automatically.'
        return True

    def load_demo(self):
        self.demo_index = (self.demo_index + 1) % 3
        t = np.linspace(0, 2 * np.pi, 600, endpoint=False)
        if self.demo_index == 0:
            radius = 175 + 47 * np.cos(5 * t)
            curve = radius * np.exp(1j * t)
        elif self.demo_index == 1:
            curve = 12 * (16 * np.sin(t) ** 3 - 1j * (
                13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)))
        else:
            curve = 190 * np.sin(3 * t) + 170j * np.sin(2 * t)
        self.load_points(curve + complex(*self.canvas.center))

    def update_trace(self):
        if self.series is not None:
            self.trace = [xy(z) for z in self.series.reconstruct(
                np.linspace(0, 1, TRACE_SAMPLES + 1), self.terms)]

    def set_terms(self, terms):
        count = max(1, min(SAMPLES, int(terms)))
        if count != self.terms:
            self.terms = count
            self.update_trace()

    def set_slider(self, x):
        fraction = (x - self.slider.left) / self.slider.width
        self.set_terms(round(1 + fraction * (SAMPLES - 1)))

    def clear(self):
        self.drawing = False
        self.dragging_slider = False
        self.series = None
        self.points = []
        self.trace = []
        self.phase = 0.0
        self.message = 'Hold the left mouse button and draw a continuous curve.'

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            key = event.key
            if key in (pygame.K_ESCAPE, pygame.K_F1):
                return False
            if key == pygame.K_SPACE:
                self.paused = not self.paused
            elif key in (pygame.K_RIGHT, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self.set_terms(self.terms + 4)
            elif key in (pygame.K_LEFT, pygame.K_MINUS, pygame.K_KP_MINUS):
                self.set_terms(self.terms - 4)
            elif key == pygame.K_UP:
                self.speed = min(4.0, self.speed * 2)
            elif key == pygame.K_DOWN:
                self.speed = max(0.25, self.speed / 2)
            elif key == pygame.K_c:
                self.show_circles = not self.show_circles
            elif key == pygame.K_o:
                self.show_original = not self.show_original
            elif key == pygame.K_r:
                self.clear()
            elif key == pygame.K_d:
                self.drawing = False
                self.load_demo()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.slider.inflate(12, 24).collidepoint(event.pos):
                self.dragging_slider = True
                self.set_slider(event.pos[0])
            elif self.canvas.collidepoint(event.pos):
                self.drawing = True
                self.points = [complex(*event.pos)]
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging_slider:
                self.set_slider(event.pos[0])
            elif self.drawing and self.canvas.collidepoint(event.pos):
                point = complex(*event.pos)
                if abs(point - self.points[-1]) >= 2 and len(self.points) < 4096:
                    self.points.append(point)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging_slider = False
            if self.drawing:
                if self.canvas.collidepoint(event.pos):
                    self.points.append(complex(*event.pos))
                self.drawing = False
                self.load_points(self.points)
        elif event.type == pygame.WINDOWFOCUSLOST:
            self.dragging_slider = False
            if self.drawing:
                self.drawing = False
                self.load_points(self.points)
        return True

    def update(self, seconds):
        if self.series is not None and not self.paused and not self.drawing:
            self.phase = (self.phase + seconds * self.speed / 8) % 1.0

    def draw_canvas(self):
        surface = self.surface
        pygame.draw.rect(surface, (14, 22, 33), self.canvas, border_radius=12)
        surface.set_clip(self.canvas)
        for x in range(self.canvas.left + 24, self.canvas.right, 32):
            for y in range(self.canvas.top + 24, self.canvas.bottom, 32):
                pygame.draw.circle(surface, (33, 45, 59), (x, y), 1)
        if self.drawing:
            if len(self.points) > 1:
                pygame.draw.aalines(surface, CYAN, False, [xy(z) for z in self.points])
        elif self.series is not None:
            if self.show_original:
                pygame.draw.aalines(surface, (57, 74, 93), True,
                                    [xy(z) for z in self.series.samples])
            chain = self.series.chain(self.phase, self.terms)
            if self.show_circles:
                for start, end in zip(chain, chain[1:]):
                    radius = abs(end - start)
                    if radius >= 1:
                        pygame.gfxdraw.aacircle(surface, *xy(start), round(radius), (67, 91, 112))
                    pygame.draw.aaline(surface, (149, 175, 193), xy(start), xy(end))
            visible = self.trace[:int(self.phase * TRACE_SAMPLES) + 1] + [xy(chain[-1])]
            if len(visible) > 1:
                pygame.draw.lines(surface, (27, 75, 79), False, visible, 6)
                pygame.draw.lines(surface, CYAN, False, visible, 2)
                pygame.draw.aalines(surface, CYAN, False, visible)
            pygame.draw.circle(surface, ORANGE, xy(chain[-1]), 5)
            pygame.draw.circle(surface, TEXT, xy(chain[-1]), 2)
        else:
            self.text('Your curve starts here.', (300, 370), MUTED)
            self.text('Hold and drag to draw.', (320, 402), MUTED, self.small)
        surface.set_clip(None)
        pygame.draw.rect(surface, BORDER, self.canvas, width=1, border_radius=12)
        self.text('CANVAS', (44, 115), MUTED, self.small)
        state = 'READY' if self.series is None else ('PAUSED' if self.paused else 'PLAYING')
        if self.drawing:
            state = 'DRAWING'
        self.text(state, (738, 115), CYAN, self.small)

    def draw_panel(self):
        surface = self.surface
        pygame.draw.rect(surface, PANEL, self.panel, border_radius=12)
        self.text('RECONSTRUCTION', (884, 120), MUTED, self.small)
        self.text(str(self.terms), (884, 164), TEXT, self.number)
        self.text('/ 256 circles', (966, 191), MUTED)
        self.text('Drag to change detail', (884, 229), MUTED, self.small)
        pygame.draw.rect(surface, BORDER, self.slider, border_radius=8)
        fraction = (self.terms - 1) / (SAMPLES - 1)
        filled = self.slider.copy()
        filled.width = max(8, round(fraction * self.slider.width))
        pygame.draw.rect(surface, CYAN, filled, border_radius=8)
        pygame.draw.circle(surface, TEXT, (round(self.slider.left + fraction * self.slider.width),
                                          self.slider.centery), 8)
        error = self.series.rms_error(self.terms) if self.series else 0
        self.text('RMS error', (884, 301), MUTED)
        self.text(f'{error:.2f} px' if self.series else '--', (1041, 301))
        self.text('Playback', (884, 334), MUTED)
        self.text(f'{self.speed:g}x', (1092, 334), ORANGE)
        pygame.draw.line(surface, BORDER, (884, 379), (1132, 379))
        self.text('LARGEST COEFFICIENTS', (884, 399), MUTED, self.small)
        if self.series is not None:
            amplitudes = np.abs(self.series.coefficients[:32])
            maximum = max(float(amplitudes.max()), 1e-9)
            for index, amplitude in enumerate(amplitudes):
                height = max(1, round(float(amplitude) / maximum * 62))
                color = CYAN if index < self.terms else BORDER
                pygame.draw.rect(surface, color, (884 + index * 7.75, 487 - height, 5, height),
                                 border_radius=2)
        self.text('Sorted by amplitude', (884, 500), MUTED, self.small)
        shortcuts = [('SPACE', 'Pause / resume'), ('LEFT / RIGHT', 'Fewer / more circles'),
                     ('UP / DOWN', 'Playback speed'), ('C / O', 'Circles / original'),
                     ('D / R', 'Next demo / clear')]
        for index, (key, action) in enumerate(shortcuts):
            y = 548 + index * 27
            self.text(key, (884, y), ORANGE, self.small)
            self.text(action, (989, y), MUTED, self.small)

    def render(self):
        self.surface.fill(BACKGROUND)
        self.text('FOURIER', (24, 25), TEXT, self.title)
        self.text('A curve, rebuilt with circles.', (209, 39), MUTED)
        self.text('256 SAMPLES  /  COMPLEX FFT', (926, 42), MUTED, self.small)
        self.draw_canvas()
        self.draw_panel()
        self.text(self.message, (24, 731), MUTED)
        self.text('ESC  Quit', (1085, 731), MUTED, self.small)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screenshot', type=Path, help='Save a demo PNG and exit without a window')
    args = parser.parse_args(argv)
    if args.screenshot:
        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    pygame.display.init()
    pygame.font.init()
    try:
        surface = pygame.Surface(SIZE) if args.screenshot else pygame.display.set_mode(SIZE)
        pygame.display.set_caption('Fourier - Epicycle Visualizer')
        app = Visualizer(surface)
        if args.screenshot:
            app.phase = 0.76
            app.render()
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(surface, str(args.screenshot))
            return
        clock = pygame.time.Clock()
        running = True
        while running:
            seconds = min(clock.tick(60) / 1000, 0.1)
            for event in pygame.event.get():
                if not app.handle_event(event):
                    running = False
                    break
            if running:
                app.update(seconds)
                app.render()
                pygame.display.flip()
    finally:
        pygame.quit()


if __name__ == '__main__':
    main()
