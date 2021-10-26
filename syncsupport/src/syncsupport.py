# -*- coding: utf-8 -*-
'''
Created on Jul 20, 2021

@author: mbergman

   This module contains support routines to assist in extracting and using PN-coded time data 
   embedded in audio files.  Portions modified from audiomezz.py (N. Frame)

'''

import numpy as np
import pyaudio
import wave 
import math
import hashlib
import struct
import argparse
import sys
from _portaudio import paInt16

#import matplotlib.pyplot as plt   # For when we want to plot intermediate curves

def nextpowerof2(value):
    if value < 2:
        return 0
    next2 = 2**(math.ceil(math.log2(value-1))) 
    return next2

def getsegmentfromtime(subjectdata, checktime, observationperiod):
    # subjectdata: an audio file with embedded PN code, from which a slice of OP length will be taken,
    # checktime: a (1/samplerate) index into subjectfile, with "time" in (1/samplerate) units starting with time t=0 at array index 0
    # observationperiod: a duration interpreted as a multiple of (1/samplerate),  
    # Returns the OP duration slice of subjectfile that begins at t=checktime in subjectfile

    if checktime + observationperiod < len(subjectdata):
        print("Error, requested window exceeds data length")
    segmentdata = subjectdata[checktime:checktime + observationperiod - 1].copy()
    return segmentdata

def gettimefromsegment(subjectdata, segmentdata):
    # Accepts 
    # 1) subjectdata: audio data in which we will search for a PN-based timestamp, 
    # 2) observationsegment: a slice of audio embedding a PN timestamp (i.e., should be from an audio file with encoded PN sequencing).  Length of observationsegment is the observation period OP. 
    # Returns
    # 1) A resulting time value tR (scalar int) in (1/samplerate) units.  The interpretation of this time value
    # is,     The value tD is referenced to t0, the beginning of the subject data, and is the delay from t0 until the first  
    #         sample of the segment appears when matched up with the subject data; the value tD is in (1/samplerate) units, 
    #         where a positive tD indicates the segment appears later than t0, and negative tD implies the segment 
    #         is cut off by trying to start before t0. 
    # This method finds the media time where the observationsegment appears in the target segmentdata.
    
    # (Timing information: the next block (to RESULTDATA) takes about 0.03s on length_result 262144.)
    # Cross-correlate the two data sets to find where the segmentdata appears in the subjectdata.
    length_result  =  nextpowerof2(len(subjectdata))
    SUBJECTDATA = np.fft.fft(subjectdata, n=length_result, norm="ortho")
    SEGMENTDATA = np.fft.fft(segmentdata, n=length_result, norm="ortho")
    RESULTDATA = np.multiply(SUBJECTDATA, np.conj(SEGMENTDATA))

    resultcomplex = (np.fft.ifft(RESULTDATA, n=length_result, norm="ortho"))
    resultdata = np.absolute(resultcomplex)
    result_tuple = np.where(resultdata == np.amax(np.absolute(resultdata)))
    result = result_tuple[0][0]

    # Plot for debug purposes
    #result_xaxis = np.arange(0, len(resultdata), 1)
    #plt.title("Correlation")
    #plt.plot(result_xaxis, resultdata)
    #plt.show()    

    return result

def check_channels(ch):
    int_ch = int(ch)
    if int_ch < 1:
        raise argparse.ArgumentTypeError("%s is an invalid number of channels, must be 1 or greater" % ch)
    return int_ch

def checkhash(filename):
    # Python program to find MD5 hash value of a file
    # Adapted from: https://www.quickprogrammingtips.com/python/how-to-calculate-md5-hash-of-a-file-in-python.html
    # Accepts
    # 1) filename
    # 2) MD5 hash string to check against; the file's current hash must match this value
    # Returns
    # 1) hashresult; True == pass, False == failed the check.

    # We assume the segmentfile is the correct one for the test but we want to check integrity.  For now, 
    # that means the calculated hash must match one of the following (we don't care which one, because 
    # we assume the test software "knows" which file to use; this is checking integrity).
    # These are Lch only version from PN build 2, should match hash via http://onlinemd5.com/
    hashPN01 = "98626CE9659298B9316A26DE2C92AE91"    
    hashPN02 = "19DEC56E552A5848CAD0350C2994A704"
    hashPN03 = "D12AB69985A3E6A7B88CA4911FC46EBA"
    hashPN04 = "BCE7DC0250E359306E3EC490416B4453"
    # Always deal with the hash as an uppercase hex string 
    hashPN01 = hashPN01.upper()
    hashPN02 = hashPN02.upper()
    hashPN03 = hashPN03.upper()
    hashPN04 = hashPN04.upper()
    
    md5_hash = hashlib.md5()
    # Read and update hash in chunks of 4K
    with open(filename,"rb") as f:
        for byte_block in iter(lambda: f.read(4096),b""):
            md5_hash.update(byte_block)
    hashresult = md5_hash.hexdigest()
    hashresult = hashresult.upper()   # Force the hash to be an uppercase hex string
    f.close()
    
    if (hashresult == hashPN01) \
            or (hashresult == hashPN02) \
            or (hashresult == hashPN03) \
            or (hashresult == hashPN04) :
        return True
    return False

def opendatafile(filename, verifyhash):
    # Accepts 
    # 1) filename: An OS file of recorded data, with or without path (must be in exec directory if without path)
    # 2) verifyhash: A boolean, should we check the file hash to verify integrity (used for the archived PN files)
    # If verifyhash is set, checks the MD5 hash of the file first.
    # Then opens and reads the file.
    # Returns a tuple of:
    # 1) samplerate, typically 48000 Hz 
    # 2) data, the array of data (frames) read from the file
    # 3) channels, the number of channels (stereo == 2)
    # 4) sampleformat, bit depth, e.g. uint16
    # 5) MD5 check result, True == pass, False == fail.

    # If this is a PN file, verify integrity before using
    hashresult = False
    if verifyhash == True:
        hashresult = checkhash(filename)

    # Open the file for reading.
    wf = wave.open(filename, 'rb')

    # Get information about the recording
    p = pyaudio.PyAudio()
    sampleformat = p.get_format_from_width(wf.getsampwidth())
    channels     = wf.getnchannels()
    samplerate   = wf.getframerate()

    # Read the data into an array.
    framecount = wf.getnframes()  # scalar
    framestring = wf.readframes(framecount)   # byte string
    unpackstring = '{0}h{0}h'.format(framecount) 
    framelist = list(struct.unpack(unpackstring, framestring)) 
    framesaschannels = np.array(framelist, dtype=np.short, ndmin=2)
    framesonlyLch = np.reshape(framesaschannels, (channels,framecount), 'F')

    p.terminate()   # End of pyaudio session

    # Return data includes channeldata but only the part that represents the L channel.
    return (samplerate, framesonlyLch, channels, sampleformat, hashresult)

def trimaudio(subjectdata, segmentdata):
    # Accepts 
    # 1) subjectdata: a first audio file that was presumably recorded with some dead time before the audio of interest, 
    # 2) segmentdata:  a segment of PN audio that should mark the end of the audio of interest, 
    # Returns: a shorter copy of subjectdata with leading audio trimmed.  The portio trimmed is the part
    # prior to first occurance of segmentdata.

    # Align the archived copy of PN data (segmentdata) with the PN data in the recorded audio (sujectdata) 
    result = gettimefromsegment(subjectdata, segmentdata[0:960])
    # Trim the front and back of the recorded audio to get rid of useless extra recording time
    trimmeddata = subjectdata[result:(len(segmentdata)+result)].copy()
 
    return trimmeddata
   
def findsegmentorder(subjectfile, segmentfile, observationperiod, neighborhood):
    # Accepts 
    # 1) subjectfile: a first (e.g.PN or recorded) audio file, possibly with extra audio before the "real" data, 
    # 2) segmentfile: a second (e.g. recorded or PN) audio file, 
    # 3) observationperiod: a duration interpreted as a multiple of (1/samplerate), 
    # 4) neighborhood: A range, in units of (1/samplerate) seconds, in which to look for a match
    # Steps through segmentfile in observationperiod increments, and repeatedly calls gettimefromsegment to find the presentation time of each increment.
    # Returns an array of presentation times.  
    
    # Open the PN sequence file (archived pseudo noise sequence in audio format), check hash, and extract the L channel
    segmentinfo = opendatafile(segmentfile, verifyhash=True)
    segmentdataPN    = segmentinfo[1][0]
    segmentrate      = segmentinfo[0]   # Should be 48000 for the WAVE project 
    segmentformat    = segmentinfo[3]   # Should be 16bit PCM
    hashresult       = segmentinfo[4]
    segmentduration   = math.floor((len(segmentdataPN))//(segmentrate))
    if segmentformat != paInt16:
        print("Error, PN file %s format is not Int16", subjectfile) 
        sys.exit(-1)   
    
    if hashresult != True:
        print("Error, PN file %s appears corrupted (failed hash check)", subjectfile) 
        sys.exit(-1)   

    # Open the subject file (recorded audio played out by the device under test) and extract the L channel
    subjectinfo    = opendatafile(subjectfile, verifyhash=False)
    rawfiledata    = subjectinfo[1][0]
    samplerate     = subjectinfo[0]
    sampleformat   = subjectinfo[3] # Can also fetch channels as subjectinfo[2] if needed
    
    # Trim off any leading (useless) audio prior to the watermarked portion
    filedata = trimaudio(rawfiledata, segmentdataPN)
    
    duration   = math.floor((len(filedata))/samplerate)
    if duration != segmentduration:
        print("Error, the two files have different durations, should be 60 seconds--possibly less than 60s of audio from start of watermark to end of file, or inadequate audio SNR") 
        sys.exit(-1)   
    if samplerate != segmentrate:
        print("Error, the two files have different sample rates, should be 48000") 
        sys.exit(-1)   
    if sampleformat != segmentformat:
        print("Error, the two files have different formats; should be 16b PCM") 
        sys.exit(-1)   
    
    maxsegments = math.floor(duration*samplerate/observationperiod)
    resultarr = np.zeros((maxsegments), dtype=np.uint) 
    for idx in range(0, maxsegments):
        # To speed things up, only check in the expected neighborhood of the segment (e.g., +/- 500mS).
        neighborstart = (idx*observationperiod)-(neighborhood//2)
        neighborend   = (idx*observationperiod)+(neighborhood//2)
        if neighborstart < 0:
            neighborstart = 0
            neighborend = neighborhood
        if neighborend > len(filedata):
            neighborstart = len(filedata) - neighborhood
            neighborend = len(filedata)
        
        subjectdata = filedata[neighborstart:neighborend].view()
        thissegment = segmentdataPN[(idx*observationperiod):((idx+1)*observationperiod)].view()
        resultraw = gettimefromsegment(subjectdata, thissegment) + neighborstart
        
        # Clean up error due to differences in playout and record sample clock rates; we assume the error is small w.r.t. the correct value. 
        resultclean = int(round(resultraw/observationperiod,0)*observationperiod)
        resultarr[idx] = resultclean 
        
    return resultarr

