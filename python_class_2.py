import numpy as np

class Book(object):
    count = 0 
    
    def __init__(self,author,title,publisher,date):
        self.author = author
        self.title = title
        self.publisher = publisher
        self.date = date
        Book.count += 1
        
    def __str__(self):
        return ('Author:'+ self.author +\
                '\ntitle :'+ self.title +\
                '\npublisher:' + self.publisher +\
                '\nDate:' + self.date)
        
book = Book('jungwon','python','colab','2020')
print(book)
print('Number of Books :', str(Book.count))

##############################################################################

class Vehicle(object):
    speed = 0
    def up_speed(self, value):
        self.speed += value
        
    def down_speed(self, value):
        self.speed -= value
    
    def print_speed(self):
        print('speed:', str(self.speed))
        
class Car(Vehicle):
    def up_speed(self, value):
        self.speed += value
        if self.speed > 240 : self.speed = 240
        
class Truck(Vehicle):
    def up_speed(self, value):
        self.speed += value
        if self.speed > 180 : self.speed = 180

car = Car()
car.up_speed(300)
car.print_speed()

truck = Truck()
truck.up_speed(180)
truck.print_speed()

##############################################################################
class Next:
    List = []

def __init__(self,low,high) :
    for Num in range(low,high) :
        self.List.append(Num ** 2)

def __call__(self,Nu):
    return self.List[Nu]

b = Next(1,7)
print (b.List)
print (b(2))