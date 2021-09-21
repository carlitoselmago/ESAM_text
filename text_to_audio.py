#INSTALLATION WITH PIP
# pip install pyo

from pyo import *
from time import sleep
from threading import Thread
import random

#init synth module
s = Server().boot()
s.start()

def play_sound(value):
    #play a sound of a synth with custom values based on text length
    freq = value*10
    lfo = Sine(freq=value, mul=.02, add=1)
    lf2 = Sine(freq=.25, mul=10, add=value)
    freqs=[]
    for r in range(random.randint(1,int(value/2))+1):
        freqs.append(value*1.5)
    a = Blit(freq=freqs*lfo, harms=lf2, mul=.3).out()
    sleep(value)

while True:
    val = input("Ask me a question: ")
    valLen=len(val) #length of value
    thread = Thread(target=play_sound,args=(valLen,))
    thread.start()

    sleep(valLen) #wait n seconds
