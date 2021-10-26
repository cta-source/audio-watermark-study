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
samplerate = 48000          # Audio sample rate  (not used here except in the comments; this rate is a hard-coded value in the content and PN files).
observationperiod = 960     # This value works out to 20mS at a 48 kHz audio sample rate.
neighborhood = 25*observationperiod  # Units are 1/RATE seconds. Checking each segment of a PN sequence against 60 seconds of audio takes too long.  Only check in the "neighborhood" of where the segment is expected to be.

# Test files
segmentfile = "./src/PN01.wav"  # Reference (a priori) watermark filename.
# Choose one of:
subjectfile = "./src/PN01.wav"                                          # Simplest test, verify the software can sync PN01 to itself.
#subjectfile = "./PN01 FS 0 s to 1s stepped -1dB per sec w Adagio 5s-6s repeat FS.wav"  # Demonstration file generated on Audacity, a 1s loop of Adagio repeats for 60s, mixed with PN01 at full scale at t=0, -1dB starting t=1s, -2dB at 2s etc.
#subjectfile = "./Recorded - basic PN01 w mic close to speakers.wav"     # Demonstration file, playout of PN01 over speakers on one PC, and recorded via mic on a second PC.
#subjectfile = "./Recorded - basic PN01 wired via USB input device.wav"  # Demonstration file, playout of PN01 on one PC, and recorded via direct wired connection (line-out to line-in) on a second PC.

print("\n    enter findsegmentorder at "+str(time.asctime(time.gmtime())))
segmenttimingarray = findsegmentorder(subjectfile, segmentfile, observationperiod, neighborhood)
print("\n    leave findsegmentorder at "+str(time.asctime(time.gmtime())))

# Check the result, satisfying "all segments in order" requirement in DPCTF spec
segmentcount = len(segmenttimingarray)
errorcount = 0
for idx in range(1, segmentcount):
    if ((idx)*observationperiod) != segmenttimingarray[idx]:
        print("Error, segments out of order, expected "+str((idx-1)*observationperiod)+", found "+str(segmenttimingarray[idx]))
        errorcount += 1
        if errorcount > 500:
            print("Error: found greater than "+str(errorcount)+" segments out of order; terminating program")
            print("\n    Finish and terminate at "+str(time.asctime(time.gmtime())))
            sys.exit(-1)

if errorcount == 0:
    print("Success -- all segments in order")
else:
    print("Error: found "+str(errorcount)+" segments out of order")
print("\n    Finish and terminate at "+str(time.asctime(time.gmtime())))
    
sys.exit(1)   
