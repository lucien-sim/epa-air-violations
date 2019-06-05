#!/usr/bin/env python3

import os
import datetime

if __name__ == '__main__': 
    
    os.remove('logfile.txt')
    with open('logfile.txt','w+') as log: 
        date = str(datetime.datetime.today())
        log.write('Beginning retraining process: '+date+'\n\n')
    
    