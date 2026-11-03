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
BACKGROUND = (247, 246, 242)
BORDER = (216, 216, 209)
TEXT = (47, 52, 53)
MUTED = (117, 121, 118)
ACCENT = (48, 94, 117)


def xy(point):
    return (round(point.real), round(point.imag))


class Visualizer:
    def __init__(self, surface):
        self.surface = surface
        self.canvas = pygame.Rect(32, 88, 1116, 552)
        self.font = pygame.font.SysFont('helveticaneue,arial,liberationsans', 18)
        self.small = pygame.font.SysFont('helveticaneue,arial,liberationsans', 15)
        self.title = pygame.font.SysFont('georgia,dejavuserif', 31)
        self.slider = pygame.Rect(222, 704, 260, 3)
        self.buttons = {
            'demo': pygame.Rect(969, 26, 83, 34),
            'clear': pygame.Rect(1070, 26, 70, 34),
            'pause': pygame.Rect(40, 687, 86, 36),
            'speed': pygame.Rect(679, 687, 57, 36),
            'circles': pygame.Rect(795, 688, 123, 34),
            'original': pygame.Rect(954, 688, 130, 34),
        }
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
        self.message = 'Drag to draw a curve.'
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
        self.message = 'Drag to draw a new curve.'
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
            self.trace = [(float(z.real), float(z.imag)) for z in self.series.reconstruct(
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
        self.message = 'Draw a curve with the mouse.'

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
            if self.buttons['demo'].collidepoint(event.pos):
                self.drawing = False
                self.load_demo()
            elif self.buttons['clear'].collidepoint(event.pos):
                self.clear()
            elif self.buttons['pause'].collidepoint(event.pos):
                self.paused = not self.paused
            elif self.buttons['speed'].collidepoint(event.pos):
                self.speed = self.speed * 2 if self.speed < 4 else 0.25
            elif self.buttons['circles'].collidepoint(event.pos):
                self.show_circles = not self.show_circles
            elif self.buttons['original'].collidepoint(event.pos):
                self.show_original = not self.show_original
            elif self.slider.inflate(12, 24).collidepoint(event.pos):
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
        surface.set_clip(self.canvas)
        if self.drawing:
            if len(self.points) > 1:
                pygame.draw.aalines(surface, ACCENT, False, [xy(z) for z in self.points])
        elif self.series is not None:
            if self.show_original:
                pygame.draw.aalines(surface, (207, 209, 201), True,
                                    [xy(z) for z in self.series.samples])
            chain = self.series.chain(self.phase, self.terms)
            if self.show_circles:
                for start, end in zip(chain, chain[1:]):
                    radius = abs(end - start)
                    if radius >= 1:
                        pygame.gfxdraw.aacircle(surface, *xy(start), round(radius), (185, 193, 192))
                    pygame.draw.aaline(surface, (135, 151, 155), xy(start), xy(end))
                center = xy(self.series.origin)
                pygame.draw.circle(surface, MUTED, center, 2)
            endpoint = (float(chain[-1].real), float(chain[-1].imag))
            visible = self.trace[:int(self.phase * TRACE_SAMPLES) + 1] + [endpoint]
            if len(visible) > 1:
                pygame.draw.aalines(surface, ACCENT, False, visible)
            pygame.gfxdraw.filled_circle(surface, *xy(chain[-1]), 4, ACCENT)
            pygame.gfxdraw.aacircle(surface, *xy(chain[-1]), 4, ACCENT)
        else:
            label = self.font.render('Draw a curve to begin', True, MUTED)
            surface.blit(label, label.get_rect(center=self.canvas.center))
        surface.set_clip(None)

    def draw_button(self, name, label):
        rect = self.buttons[name]
        if rect.collidepoint(pygame.mouse.get_pos()):
            pygame.draw.rect(self.surface, (235, 235, 229), rect, border_radius=4)
        rendered = self.font.render(label, True, TEXT)
        self.surface.blit(rendered, rendered.get_rect(center=rect.center))

    def draw_checkbox(self, name, label, checked):
        rect = self.buttons[name]
        box = pygame.Rect(rect.left, rect.centery - 7, 14, 14)
        pygame.draw.rect(self.surface, ACCENT if checked else BORDER, box,
                         width=0 if checked else 1, border_radius=2)
        if checked:
            pygame.draw.lines(self.surface, BACKGROUND, False,
                              [(box.x + 3, box.y + 7), (box.x + 6, box.y + 10),
                               (box.x + 11, box.y + 4)], 2)
        rendered = self.font.render(label, True, TEXT)
        self.surface.blit(rendered, (rect.left + 24, rect.centery - rendered.get_height() // 2))

    def draw_controls(self):
        surface = self.surface
        pygame.draw.line(surface, BORDER, (40, 660), (1140, 660))
        self.draw_button('pause', 'Play' if self.paused else 'Pause')
        self.text('Detail', (156, 693), MUTED)
        pygame.draw.rect(surface, BORDER, self.slider)
        fraction = (self.terms - 1) / (SAMPLES - 1)
        position = round(self.slider.left + fraction * self.slider.width)
        pygame.draw.line(surface, ACCENT, self.slider.midleft, (position, self.slider.centery), 3)
        pygame.gfxdraw.filled_circle(surface, position, self.slider.centery, 6, ACCENT)
        pygame.gfxdraw.aacircle(surface, position, self.slider.centery, 6, ACCENT)
        self.text(f'{self.terms} circles', (503, 693))
        self.text('Speed', (620, 693), MUTED)
        self.draw_button('speed', f'{self.speed:g}x')
        self.draw_checkbox('circles', 'Circles', self.show_circles)
        self.draw_checkbox('original', 'Original', self.show_original)
        self.text('Space  pause     /     Left & Right  detail     /     D  example',
                  (40, 746), MUTED, self.small)
        if self.series is not None and not self.drawing:
            error = self.small.render(f'RMS error  {self.series.rms_error(self.terms):.2f} px', True, MUTED)
            surface.blit(error, (1140 - error.get_width(), 746))

    def render(self):
        self.surface.fill(BACKGROUND)
        self.text('Fourier', (40, 24), TEXT, self.title)
        message = 'Release to trace your curve.' if self.drawing else self.message
        hint = self.small.render(message, True, MUTED)
        self.surface.blit(hint, hint.get_rect(center=(SIZE[0] // 2, 43)))
        self.draw_button('demo', 'Example')
        self.draw_button('clear', 'Clear')
        self.draw_canvas()
        self.draw_controls()


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
