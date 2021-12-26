import pygame
import random
from time import sleep


WINDOW_WIDTH = 480
WINDOW_HEIGHT = 800

BLACK = (0,0,0)
WHITE = (255,255,255)
GRAY = (150,150,150)
RED = (255,0,0)

class Car :
    image_car = ['RacingCar01.png','RacingCar02.png','RacingCar03.png','RacingCar04.png','RacingCar05.png',\
                 'RacingCar06.png','RacingCar07.png','RacingCar08.png','RacingCar09.png','RacingCar10.png',\
                 'RacingCar11.png','RacingCar12.png','RacingCar13.png','RacingCar14.png','RacingCar15.png',\
                 'RacingCar16.png','RacingCar17.png','RacingCar18.png','RacingCar19.png','RacingCar20.png',]
    def __init__(self, x=0, y=0,dx=0,dy = 0):
        self.image = ''
        self.x = x 
        self.y = y
        self.dx = dx
        self.dy = dy
        
    def load_image(self):
        self.image = pygame.image.load(random.choice(self.image_car))
        self.width = self.image.get_rect().size[0]
        self.height = self.image.get_rect().size[1]
        
    def draw_image(self):
        screen.blit(self.image,[self.x,self.y])
        
    def move_x(self):
        self.x += self.dx
        
    def move_y(self):
        self.y += self.dy
    
    def check_out_of_screen(self):
        if self.x+self.width > WINDOW_WIDTH or self.x < 0:
            self.x -= self.dx
    def check_crash(self, car):
        if (self.x+self.width > car.x) and (self.x < car.x+car.width) and (self.y < car.y+car.height) and (self.y+self.height > car.y):
            return True
        else:
            return False
        
if __name__ == '__main__': 
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption('pycar : Racing Car Game')
    clock = pygame.time.Clock()
    
    pygame.mixer.music.load('race.wav')
    sound_crash = pygame.mixer.Sound('crash.wav')
    sound_engine = pygame.mixer.Sound('engine.wav')
    
    player = Car((WINDOW_WIDTH / 2),(WINDOW_HEIGHT - 150),0,0)
    player.load_image()
    
    cars = []
    car_count = 3
    for i in range(car_count):
        x = random.randrange(0, WINDOW_WIDTH - 55)
        y = random.randrange(-15,-50, -35)
        car = Car(x,y, 0, random.randint(5,10)) 
        car.load_image()
        cars.append(car)
        

                                                                                                           