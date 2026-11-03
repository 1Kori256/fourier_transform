import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pygame

from main import SIZE, Visualizer, main


class AppTests(unittest.TestCase):
    def setUp(self):
        pygame.display.init()
        pygame.font.init()
        self.app = Visualizer(pygame.Surface(SIZE))

    def tearDown(self):
        pygame.quit()

    def key(self, key):
        return self.app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key))

    def test_playback_uses_elapsed_time_and_pause(self):
        self.app.update(1)
        self.assertAlmostEqual(self.app.phase, 0.125)
        self.key(pygame.K_SPACE)
        self.app.update(1)
        self.assertAlmostEqual(self.app.phase, 0.125)
        self.key(pygame.K_SPACE)
        self.key(pygame.K_UP)
        self.app.update(1)
        self.assertAlmostEqual(self.app.phase, 0.375)

    def test_equivalent_frame_rates_produce_same_phase(self):
        for _ in range(30):
            self.app.update(1 / 30)
        first = self.app.phase
        self.app.phase = 0
        for _ in range(144):
            self.app.update(1 / 144)
        self.assertAlmostEqual(self.app.phase, first)

    def test_slider_clamps_terms_and_rebuilds_trace(self):
        self.app.set_slider(-100)
        self.assertEqual(self.app.terms, 1)
        self.app.set_slider(10000)
        self.assertEqual(self.app.terms, 256)
        self.assertEqual(len(self.app.trace), 1025)

    def test_click_without_stroke_keeps_previous_curve(self):
        previous = self.app.series
        for event_type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            self.app.handle_event(pygame.event.Event(event_type, pos=(300, 300), button=1))
        self.assertIs(self.app.series, previous)
        self.assertFalse(self.app.drawing)
        self.app.render()

    def test_clear_during_drawing_does_not_leave_an_invalid_stroke(self):
        self.app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                               pos=(300, 300), button=1))
        self.key(pygame.K_r)
        self.app.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=(320, 320)))
        self.assertFalse(self.app.drawing)
        self.assertIsNone(self.app.series)
        self.app.render()

    def test_draw_release_and_focus_loss(self):
        for end_event in (pygame.MOUSEBUTTONUP, pygame.WINDOWFOCUSLOST):
            with self.subTest(end_event=end_event):
                self.app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                                       pos=(300, 300), button=1))
                self.app.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=(400, 300)))
                self.app.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=(350, 400)))
                self.app.handle_event(pygame.event.Event(end_event, pos=(300, 300), button=1))
                self.assertFalse(self.app.drawing)
                self.assertEqual(len(self.app.series.samples), 256)
                self.app.render()

    def test_all_demos_and_visibility_controls_render(self):
        self.key(pygame.K_c)
        self.key(pygame.K_o)
        for _ in range(3):
            self.key(pygame.K_d)
            self.app.update(2)
            self.app.render()
        self.assertFalse(self.app.show_circles)
        self.assertFalse(self.app.show_original)
        self.assertFalse(self.key(pygame.K_ESCAPE))

    def test_screenshot_cli_writes_valid_png(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'preview.png'
            result = subprocess.run([sys.executable, 'main.py', '--screenshot', str(path)],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(pygame.image.load(str(path)).get_size(), SIZE)

    def test_main_loop_exits_cleanly_on_quit_event(self):
        # Avoid a window: use SDL's dummy driver and queue quit before the loop.
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        main([])
        self.assertFalse(pygame.display.get_init())


if __name__ == '__main__':
    unittest.main()
