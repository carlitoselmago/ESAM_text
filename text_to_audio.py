from pyo import *
from time import sleep
from threading import Thread

s = Server().boot()
s.start()


def play_sound(value):
    # Sets fundamental frequency.

    freq = value*10#187.5
    lfo = Sine(freq=value, mul=.02, add=1)
    lf2 = Sine(freq=.25, mul=10, add=value)
    a = Blit(freq=[100, 99.7]*lfo, harms=lf2, mul=.3).out()
    sleep(value)




while True:
    val = input("Ask me a question: ")
    #print(val)
    valLen=len(val)
    thread = Thread(target=play_sound,args=(valLen,))
    thread.start()

    sleep(valLen)
