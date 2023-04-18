from math import *

import math
import pygame
import time as time2

pygame.init()
pygame.display.set_caption('Fourier Transform')

WIDTH = 800
HEIGHT = 800

win = pygame.display.set_mode((WIDTH, HEIGHT))


class Complex():

    def __init__(self, real_part, imaginary_part):
        self.re = real_part
        self.im = imaginary_part

    def __add__(self, other):
        return Complex(self.re + other.re, self.im + other.im)

    def __mul__(self, other):
        return Complex(self.re * other.re - self.im * other.im, self.re * other.im + self.im * other.re)

    def __repr__(self):
        return f"{self.re} + {self.im}i"


def discrete_fourier_transform(x):
    X = []
    N = len(x)
    for k in range(0, N):
        s = Complex(0, 0)
        for n in range(0, N):
            phi = (2 * pi * k * n) / N
            c = Complex(cos(phi), sin(phi))
            s = s + (c * x[n])

        s.re = s.re / N
        s.im = s.im / N

        frequency = k
        radius = sqrt(s.re * s.re + s.im * s.im)
        phase = atan2(s.im, s.re)

        X.append([s.re, s.im, frequency, radius, phase])

    return X


def epicycles(x, y, rotation, fourier, time, surface):
    x_list = [x]
    y_list = [y]

    N = len(fourier)
    for i in range(0, N):
        previuos_x = x
        previous_y = y
        frequency = fourier[i][2]
        radius = fourier[i][3]
        phase = fourier[i][4]
        x += radius * cos(frequency * time + phase + rotation)
        y += radius * sin(frequency * time + phase + rotation)

        x_list.append(x)
        y_list.append(y)

        pygame.draw.circle(surface, (50, 50, 50), (previuos_x, previous_y), radius, width=3)

    for i in range(1, N):
        pygame.draw.line(surface, (255, 255, 255), (x_list[i - 1], y_list[i - 1]), (x_list[i], y_list[i]), 2)

    return (x, y)


def sorting_parameter(e):
    return e[3]


def main(surface):
    state = -1
    time = 0
    x = []
    p = []
    pshow = []
    path = []

    run = True
    while run:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                run = False
                pygame.display.quit()

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_F1:
                    run = False
                    pygame.display.quit()

            if event.type == pygame.MOUSEMOTION:
                mouse_pos = pygame.mouse.get_pos()

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if event.button == 1:
                    state = "INPUT"
                    time = 0
                    x = []
                    p = []
                    path = []
                    pshow = []

            if event.type == pygame.MOUSEBUTTONUP:
                mouse_pos = pygame.mouse.get_pos()

                if event.button == 1:
                    state = "OUTPUT"

                    if len(p) >= 500:
                        skip = int(len(p) / 500)
                    else:
                        skip = 1
                    for point in range(0, len(p), skip):
                        x.append(Complex(p[point][0], p[point][1]))

                    fourierX = discrete_fourier_transform(x)

                    fourierX.sort(key=sorting_parameter, reverse=True)

        surface.fill((0, 0, 0))

        if state == "INPUT":
            pshow.append(mouse_pos)
            p.append((mouse_pos[0] - WIDTH / 2, mouse_pos[1] - HEIGHT / 2))

            if len(pshow) >= 2:
                pygame.draw.lines(surface, (255, 255, 255), False, pshow, 3)

        if state == "OUTPUT":

            v = epicycles(WIDTH / 2, HEIGHT / 2, 0, fourierX, time, surface)
            path.insert(0, v)

            if len(path) >= 3:
                pygame.draw.lines(surface, (255, 0, 0), False, path[:-1], 3)

            try:
                time += 2 * pi / len(fourierX)
            except:
                pass

            if time > 2 * pi:
                time = 0
                path = []

            #time2.sleep(0.1)

        pygame.display.update()


main(win)
pygame.quit()