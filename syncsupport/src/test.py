# -*- coding: utf-8 -*-
'''
Created on Jul 27, 2021

@author: mbergman

Demonstration program using the syncsupport methods.

'''
import sys
import time
from syncsupport import findsegmentorder

# Default parameters
# RATE = 48000              # Audio sample rate  (not used here except in the comments; this rate is a hard-coded value in syncsupport.
observationperiod = 960    # This value works out to 80mS.  Target should be 960 (20mS).  However, the signal-to-noise in the test file is too great for 20mS to work.
cyclicpad = 0               # (units are 1/RATE seconds) For signal conditioning, usually some number of audio samples at 48kHz sample rate. "0" means "not used".  For short (20-80mS) observation period, not helpful so keep at 0.
neighborhood = 48000        # (units are 1/RATE seconds) Checking each segment of a PN sequence against 60 seconds of audio takes too long.  Only check in the "neighborhood" of where the segment is expected to be.

# Test files
subjectfile = "./src/AdagioPN01-12dB2snoise1.wav"  # Demonstration file with 2 seconds of random noise, 60 seconds of an adagio with PN01 noise, and 2 seconds of random noise.
segmentfile = "./src/PN01.wav"

print("\n    enter findsegmentorder at "+str(time.asctime(time.gmtime())))
segmenttimingarray = findsegmentorder(subjectfile, segmentfile, observationperiod, cyclicpad, neighborhood)
print("\n    leave findsegmentorder at "+str(time.asctime(time.gmtime())))

# Check the result, satisfying "all segments in order" requirement in DPCTF spec
segmentcount = len(segmenttimingarray)
errorcount = 0
for idx in range(1, segmentcount):
    if segmenttimingarray[idx] <= segmenttimingarray[idx-1]:
        print("Error, segments out of order, expected "+str((idx-1)*observationperiod)+", found "+str(segmenttimingarray[idx]))
        errorcount += 1
        #sys.exit(-1)

if errorcount == 0:
    print("Success -- all segments in order")
else:
    print("Error: found "+str(errorcount)+" segments out of order")
    
sys.exit(1)   

