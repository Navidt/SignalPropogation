from sense_hat import SenseHat
from time import sleep
from matplotlib import pyplot as plt
import numpy as np


sense = SenseHat()
sense.clear()

t = sense.get_temperature()
p = sense.get_pressure()
h = sense.get_humidity()
# temps = deque()
temps = []
all_t = []
avg_temps = []


#while(1):
for i in range(5000):
    temps.append(sense.get_temperature())
    all_t.append(sense.get_temperature())
    
    if(len(temps) >= 200):
        
        avg_temp = sum(temps) / len(temps)
        avg_temps.append(avg_temp)
        temps.pop(0)

        
    

print(all_t)
print(avg_temps)
plt.plot(avg_temps)
plt.plot(all_t)
plt.show()
    
