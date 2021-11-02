import os
import time

i = 0
while True:
    i += 1
    os.system('echo ' + '-'*20 + 'CPU' + '-'*20 + str(i))
    os.system('top -bn1|egrep "(%CPU)|([.].+:)"|grep -v root|grep -v " 0.0 "|tail -9')
    os.system('~/anaconda3/envs/tf/bin/gpustat -cpu --color|grep :')
    time.sleep(5)
    os.system('echo')
