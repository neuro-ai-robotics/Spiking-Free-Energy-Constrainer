#Imports
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()
import pygame
from pygame.locals import *

#Setting up FPS 
FPS = 1000  

#Creating colors
RED   = (255, 0, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

#Other Variables for use in the program
SCREEN_HEIGHT = 1000
SCREEN_WIDTH = int(1.5*SCREEN_HEIGHT)

unit = SCREEN_HEIGHT//20

class SpringMassDamper_Display(pygame.sprite.Group):
    class Wall(pygame.sprite.Sprite):
        def __init__(self, group=None):
            super().__init__(group)
            self.width = 50
            self.height = SCREEN_HEIGHT
            self.image = pygame.Surface((self.width, self.height))
            self.image.fill(BLACK)
            self.rect = self.image.get_rect()
            self.rect.topleft = (0, 0)

    class Mass(pygame.sprite.Sprite):
        def __init__(self, width, height, color, group=None):
            super().__init__(group)
            self.width = width
            self.height = height
            self.color = color
            self.image = pygame.Surface((width, height))
            self.image.fill(color)
            self.rect = self.image.get_rect()
            # Set an initial position; it will be updated in update()
            self.rect.center = (600, SCREEN_HEIGHT // 2)
            self.x_position = 600

        def update(self):
            # Update mass position horizontally; center it vertically.
            self.rect.center = (int(self.x_position), SCREEN_HEIGHT // 2)

    class TextPart(pygame.sprite.Sprite):
        def __init__(self, group=None):
            super().__init__(group)
            self.font = pygame.font.Font(None, 100)
            self.time = 0
            self.image = self.font.render("Time: 0.00s", True, BLACK)
            self.rect = self.image.get_rect(top=20, left=SCREEN_WIDTH - 500)

        def update(self):
            self.image = self.font.render("Time: " + str(np.round(self.time, 2)) + "s", True, BLACK)
            self.rect = self.image.get_rect(top=20, left=SCREEN_WIDTH - 500)

    def __init__(self):
        super().__init__()
        # Create the wall, mass, and text sprites.
        self.wall = self.Wall(self)
        self.mass = self.Mass(100, 50, RED, self)
        self.text_part = self.TextPart(self)
        # Equilibrium x-position for the mass when displacement = 0.
        self.equilibrium = 600  
        # Scaling factor to convert system displacement (state[0]) to pixels.
        self.scale = unit  # You already defined unit in your script.
        # Fraction of the distance (from wall to mass) drawn as the spring.
        self.spring_fraction = 0.7

    def update(self, state, time):
        # Assume state[0] is the horizontal displacement from equilibrium.
        displacement = state[0]
        # Compute the mass x position: equilibrium plus scaled displacement.
        mass_x = self.equilibrium + displacement * self.scale
        
        self.mass.x_position = mass_x
        self.text_part.time = time
        super().update()

    def draw(self, surface):
        # Draw the wall.
        surface.blit(self.wall.image, self.wall.rect)
        # Draw the spring-damper connecting the wall to the mass.
        # Starting point for the spring is the right edge of the wall.
        start_point = (self.wall.rect.right, SCREEN_HEIGHT // 2)
        # Ending point is taken as the left edge of the mass.
        end_point = (self.mass.rect.left, self.mass.rect.centery)
        total_length = end_point[0] - start_point[0]
        if total_length < 0:
            total_length = 0
        # Partition the distance into spring and damper segments.
        spring_length = total_length * self.spring_fraction
        damper_length = total_length - spring_length
        # The spring will extend from start_point to spring_end.
        spring_end = (start_point[0] + spring_length, start_point[1])
        
        # Draw the spring as a zigzag line.
        num_zigs = 10  # number of segments for the zigzag
        spring_points = []
        for i in range(num_zigs + 1):
            t = i / num_zigs
            x = start_point[0] + t * spring_length
            # For interior points, alternate vertical offsets.
            if i == 0 or i == num_zigs:
                y = start_point[1]
            else:
                direction = -1 if i % 2 == 0 else 1
                amplitude = 10  # adjust amplitude as desired
                y = start_point[1] + direction * amplitude
            spring_points.append((int(x), int(y)))
        if len(spring_points) > 1:
            pygame.draw.lines(surface, BLACK, False, spring_points, 3)
        
        # Draw the damper as a rectangle.
        damper_height = 30
        damper_top = start_point[1] - damper_height // 2
        damper_rect = pygame.Rect(spring_end[0], damper_top, int(damper_length), damper_height)
        pygame.draw.rect(surface, BLACK, damper_rect, 3)  # outline of the damper
        
        # Optionally, connect the spring end to the damper and damper to the mass.
        pygame.draw.line(surface, BLACK, spring_end, (damper_rect.left, start_point[1]), 3)
        pygame.draw.line(surface, BLACK, (damper_rect.right, start_point[1]), end_point, 3)
        
        # Finally, draw the remaining sprites (mass and time text).
        super().draw(surface)

class MassChain_Display(pygame.sprite.Group):
    class Wall(pygame.sprite.Sprite):
        def __init__(self, pos, size, group=None):
            super().__init__(group)
            self.image = pygame.Surface(size)
            self.image.fill(BLACK)
            self.rect = self.image.get_rect(topleft=pos)
    
    class Mass(pygame.sprite.Sprite):
        def __init__(self, width, height, color, group=None):
            super().__init__(group)
            self.width = width
            self.height = height
            self.color = color
            self.image = pygame.Surface((width, height))
            self.image.fill(color)
            self.rect = self.image.get_rect()
            self.x_position = SCREEN_WIDTH // 2
            self.y_position = SCREEN_HEIGHT // 2
        
        def update(self):
            self.rect.center = (int(self.x_position), int(self.y_position))
    
    class TextPart(pygame.sprite.Sprite):
        def __init__(self, group=None):
            super().__init__(group)
            self.font = pygame.font.Font(None, 100)
            self.time = 0
            self.image = self.font.render("Time: 0.00s", True, BLACK)
            self.rect = self.image.get_rect(top=20, left=SCREEN_WIDTH - 500)
        
        def update(self):
            self.image = self.font.render("Time: " + str(np.round(self.time, 2)) + "s", True, BLACK)
            self.rect = self.image.get_rect(top=20, left=SCREEN_WIDTH - 500)

    def __init__(self, N=5):
        """
        N: Number of masses in the chain.
        """
        super().__init__()
        self.N = N
        
        # Create static walls.
        wall_width = 50
        wall_height = SCREEN_HEIGHT
        self.left_wall = self.Wall((0, 0), (wall_width, wall_height), self)
        self.right_wall = self.Wall((SCREEN_WIDTH - wall_width, 0), (wall_width, wall_height), self)
        
        # Define the horizontal region in which the masses can move:
        self.left_bound = self.left_wall.rect.right
        self.right_bound = self.right_wall.rect.left
        available_width = self.right_bound - self.left_bound
        
        # Compute equilibrium positions for each mass (evenly spaced between the walls)
        self.equilibrium_positions = []
        for i in range(N):
            x_eq = self.left_bound + (i + 1) * available_width / (N + 1)
            self.equilibrium_positions.append(x_eq)
        
        # Scaling factor to convert simulation displacement to pixels.
        #self.scale = unit  # reusing 'unit' from your script
        
        # Create mass sprites. Here each mass is drawn as a red block (50x50 pixels).
        self.masses = []
        for i in range(N):
            mass_sprite = self.Mass(50, 50, RED, self)
            mass_sprite.x_position = self.equilibrium_positions[i]
            self.masses.append(mass_sprite)
        
        # Create a text sprite to display simulation time.
        self.text_part = self.TextPart(self)
    
    def update(self, state, time):
        """
        state: A list (or array) of length N containing the displacement of each mass.
               The displacement is relative to the equilibrium position.
        time: Current simulation time.
        """
        displacement = state[0:self.N]
        for i in range(self.N):
            # New x position = equilibrium position + (displacement scaled to pixels).
            x_new = displacement[i] * 150
            self.masses[i].x_position = x_new

        self.text_part.time = time
        super().update()
    
    def _draw_spring(self, surface, start, end):
        """
        Draws a horizontal zigzag spring between two points.
        start, end: Tuples (x, y). The line is assumed to be horizontal.
        """
        num_zigs = 10
        amplitude = 10
        spring_points = []
        for i in range(num_zigs + 1):
            t = i / num_zigs
            x = start[0] + t * (end[0] - start[0])
            if i == 0 or i == num_zigs:
                y = start[1]
            else:
                y = start[1] + amplitude * ((-1) ** i)
            spring_points.append((int(x), int(y)))
        pygame.draw.lines(surface, BLACK, False, spring_points, 3)
    
    def draw(self, surface):
        # Draw the static walls.
        surface.blit(self.left_wall.image, self.left_wall.rect)
        surface.blit(self.right_wall.image, self.right_wall.rect)
        
        y_level = SCREEN_HEIGHT // 2
        
        # Draw spring between left wall and the first mass.
        start = (self.left_wall.rect.right, y_level)
        end = (self.masses[0].rect.left, y_level)
        self._draw_spring(surface, start, end)
        
        # Draw springs between consecutive masses.
        for i in range(self.N - 1):
            start = (self.masses[i].rect.right, y_level)
            end = (self.masses[i + 1].rect.left, y_level)
            self._draw_spring(surface, start, end)
        
        # Draw spring between the last mass and the right wall.
        start = (self.masses[-1].rect.right, y_level)
        end = (self.right_wall.rect.left, y_level)
        self._draw_spring(surface, start, end)
        
        # Finally, draw all the mass sprites and the time text.
        super().draw(surface)

class Masses_2D_Display(pygame.sprite.Group):
    class Drone(pygame.sprite.Sprite):
        def __init__(self, radius, color, group=None):
            super().__init__(group)
            self.radius = radius
            self.color = color
            # Create a surface with per-pixel alpha transparency.
            self.image = pygame.Surface((2 * radius, 2 * radius), pygame.SRCALPHA)
            self.rect = self.image.get_rect()
            # Draw the drone as a circle.
            pygame.draw.circle(self.image, color, (radius, radius), radius)
            self.pos = [0, 0]
        
        def update(self):
            # Update the sprite's position on screen.
            self.rect.center = (int(self.pos[0]), int(self.pos[1]))

    class TextPart(pygame.sprite.Sprite):
        def __init__(self, group=None):
            super().__init__(group)
            self.font = pygame.font.Font(None, 100)
            self.time = 0
            self.image = self.font.render("Time: 0.00s", True, BLACK)
            self.rect = self.image.get_rect(top=20, left=SCREEN_WIDTH - 500)
        
        def update(self):
            self.image = self.font.render("Time: " + str(np.round(self.time, 2)) + "s", True, BLACK)
            self.rect = self.image.get_rect(top=20, left=SCREEN_WIDTH - 500)

    def __init__(self, num_drones=3):
        """
        num_drones: Number of drones to display.
        scale: Scaling factor for the drone positions.
        """
        super().__init__()
        self.num_drones = num_drones
        self.scale = 5*unit
        # Offset so that the formation is centered on the screen.
        self.offset = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        self.drones = []
        # Create drone sprites.
        for i in range(num_drones):
            drone = self.Drone(radius=10, color=RED, group=self)
            self.drones.append(drone)
        # Create a text sprite to display simulation time.
        self.text_part = self.TextPart(self)

    def update(self, state, time):
        """
        state: A list or array of 2D positions for the drones.
               It should contain 2*num_drones elements, where for each drone:
                 - state[2*i] is the x coordinate,
                 - state[2*i+1] is the y coordinate.
        time: Current simulation time.
        """
        for i in range(self.num_drones):
            # Compute screen positions using the offset and scale.
            x = self.offset[0] + state[2 * i] * self.scale
            y = self.offset[1] + state[2 * i + 1] * self.scale
            self.drones[i].pos = (x, y)

        self.text_part.time = time
        super().update()

    def draw(self, surface):
        # Optionally, draw a grid to represent the ground plane.
        grid_color = (200, 200, 200)
        grid_spacing = 50
        for x in range(0, SCREEN_WIDTH, grid_spacing):
            pygame.draw.line(surface, grid_color, (x, 0), (x, SCREEN_HEIGHT), 1)
        for y in range(0, SCREEN_HEIGHT, grid_spacing):
            pygame.draw.line(surface, grid_color, (0, y), (SCREEN_WIDTH, y), 1)
        super().draw(surface)

