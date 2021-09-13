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

# Signal conditioning window types.  
NONE_TYPE = 0
BARTLETT_TYPE = 1
BLACKMAN_TYPE = 2
HAMMING_TYPE = 3
HANNING_TYPE = 4
RAISED_COSINE_TYPE = 4
KAISER_TYPE = 5 # Currently not used, if we want it need to add the beta param

def nextpowerof2(value):
    if value < 2:
        return 0
    next2 = 2**(math.ceil(math.log2(value-1))) 
    return next2

def applywindow(audiosegment, cyclicpad, windowtype):
    # Accepts
    # 1) a segment of audio; 
    # 2) an integer cyclicpad, units of (1/samplerate) samples (default � audio segment length; must be no longer than the audio segment length else error)
    # 3) A window type field (0 = no window; 1 = Hann/raised cosine; other values currently unused).
    # Pads the original segment: for array So[0..N-1], creates a new segment array Sp[ ] 
    # such that Sp[0..cyclicpad-1] = So[N-cp..N-1]; Sp[N..N+cyclipad-1] = So[0..cyclicpad-1]
    # Applies the specified window function with taper length cyclicpad

    # Check to see if we are not using windowing.  
    # Windowing doesn't help for very short segments (e.g. 20mS), so it is not used for most tests.
    if windowtype == NONE_TYPE:
        return audiosegment

    # Pad the passed audiosegment 
    windowdata = audiosegment[(len(audiosegment)-cyclicpad):]
    windowdata = np.append(windowdata, audiosegment)
    windowdata = np.append(windowdata, audiosegment[0:cyclicpad-1])  
    windowdata = audiosegment

    # Retrieve the selected window function values
    if windowtype == HANNING_TYPE:
        windowvalues = np.hanning(len(audiosegment))
    elif windowtype == BARTLETT_TYPE:
        windowvalues = np.bartlett(len(audiosegment))
    elif windowtype == BLACKMAN_TYPE:
        windowvalues = np.blackman(len(audiosegment))
    elif windowtype == HAMMING_TYPE:
        windowvalues = np.hamming(len(audiosegment))
    elif windowtype == KAISER_TYPE:
        print("Warning: Kaiser filter is not implemented; falling back to Hanning.")
        windowvalues = np.hanning(len(audiosegment))

    # Apply the selected window to the padded data
    # todo: windowdata is int16, windowvalues are floats.  Upconvert windowdata, multiply, scale back to int16.
    #np.multiply(windowdata[:cyclicpad-1], windowvalues[:cyclicpad-1])
    #np.multiply(windowdata[cyclicpad + len(audiosegment):], windowvalues[cyclicpad+1:]) # Note the '+1' is to skip the middle value
    windowdata = np.multiply(audiosegment, windowvalues) 
        
    return windowdata

def getsegmentfromtime(subjectdata, checktime, observationperiod):
    # subjectdata: an audio file with embedded PN code, from which a slice of OP length will be taken,
    # checktime: a (1/samplerate) index into subjectfile, with "time" in (1/samplerate) units starting with time t=0 at array index 0
    # observationperiod: a duration interpreted as a multiple of (1/samplerate),  
    # Returns the OP duration slice of subjectfile that begins at t=checktime in subjectfile

    if checktime + observationperiod < len(subjectdata):
        print("Error, requested window exceeds data length")
    segmentdata = subjectdata[checktime:checktime + observationperiod - 1].copy()
    return segmentdata

def gettimefromsegment(subjectdata, observationsegment, cyclicpad, windowtype):
    # Accepts 
    # 1) subjectdata: audio data in which we will search for a PN-based timestamp, 
    # 2) observationsegment: a slice of audio embedding a PN timestamp (i.e., should be from an audio file with encoded PN sequencing).  Length of observationsegment is the observation period OP. 
    # 3) cyclicpad: A cyclic prefix length for windowing
    # 4) windowtype: Window function type for applywindow, with:
    #   BARTLETT_TYPE = 0; BLACKMAN_TYPE = 1; HAMMING_TYPE = 2; HANNING_TYPE = 3; 
    #   RAISED_COSINE_TYPE = 3; KAISER_TYPE = 4 # Currently not used, if we want it need to add the beta param
    # Returns
    # 1) A resulting time value tR (scalar int) in (1/samplerate) units.  The interpretation of this time value
    # is,     The value tD is referenced to t0, the beginning of the subject data, and is the delay from t0 until the first  
    #         sample of the segment appears when matched up with the subject data; the value tD is in (1/samplerate) units, 
    #         where a positive tD indicates the segment appears later than t0, and negative tD implies the segment 
    #         is cut off by trying to start before t0. 
    # This method applies applywindow() to observationsegment to prepare (pad and window) the observationsegment for the FFT.  Finds
    # the presentation time in subjectfile in which the windowed observationsegment appears; and returns the presentation 
    # time of the beginning of the original observationsegment (not counting the cyclic prefix) in subjectfile.

    # Apply the window function.  If cyclicpad == 0, don't apply cyclic padding.  If windowtype == NONE_TYPE, don't apply windowing.
    segmentdata = applywindow(observationsegment, cyclicpad, windowtype)
    
    # Cross-correlate the two data sets to find where the segmentdata appears in the subjectdata.
    length_result  =  nextpowerof2(len(subjectdata))
    SUBJECTDATA = np.fft.fft(subjectdata, n=length_result, norm="ortho")
    SEGMENTDATA = np.fft.fft(segmentdata, n=length_result, norm="ortho")
    RESULTDATA = np.multiply(SUBJECTDATA, np.conj(SEGMENTDATA))

    resultcomplex = (np.fft.ifft(RESULTDATA, n=length_result, norm="ortho"))
    resultdata = resultcomplex.real
    result_tuple = np.where(resultdata == np.amax(resultdata))
    result = result_tuple[0]

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

    # Read the data into an array 
    myframes    = wf.getnframes()  # scalar
    myframesbig = wf.readframes(myframes)   # byte string

    unpackstring = '<{0}h'.format(myframes*channels) 
    channeldata = list(struct.unpack(unpackstring, myframesbig)) # Convert the byte string into a list of ints
 
    p.terminate()   # End of pyaudio session

    # Return data includes channeldata but only the part that represents the L channel.
    return (samplerate, channeldata[0::channels], channels, sampleformat, hashresult)

def trimaudio(subjectdata, segmentdata, cyclicpad, windowtype):
    # Accepts 
    # 1) subjectdata: a first audio file that was presumably recorded with some dead time before the audio of interest, 
    # 2) segmentdata:  a segment of PN audio that should mark the end of the audio of interest, 
    # 3) cyclicpad: A cyclic prefix length for windowing
    # 4) windowtype: Window function type for applywindow
    # Returns: a shorter copy of subjectdata with leading audio trimmed.  The portio trimmed is the part
    # prior to first occurance of segmentdata.

    # Align the archived copy of PN data (segmentdata) with the PN data in the recorded audio (sujectdata) 
    result = gettimefromsegment(subjectdata, segmentdata, cyclicpad, windowtype)
    # Trim the front and back of the recorded audio to get rid of useless extra recording time
    trimmeddata = subjectdata[result[0]:(len(segmentdata)+result[0])].copy()
 
    return trimmeddata

def findsegmentorder(subjectfile, segmentfile, observationperiod, cyclicpad, neighborhood):
    # Accepts 
    # 1) subjectfile: a first (e.g.PN or recorded) audio file, possibly with extra audio before the "real" data, 
    # 2) segmentfile: a second (e.g. recorded or PN) audio file, 
    # 3) observationperiod: a duration interpreted as a multiple of (1/samplerate), 
    # 4) cyclicpad: A cyclic prefix length for windowing
    # 5) neighborhood: A range, in units of (1/samplerate) seconds, in which to look for a match
    # Steps through segmentfile in observationperiod increments, and repeatedly calls gettimefromsegment to find the presentation time of each increment.
    # Returns an array of presentation times.  

    # Open the PN sequence file (archived pseudo noise sequence in audio format), check hash, and extract the L channel
    segmentinfo = opendatafile(segmentfile, verifyhash=True)
    segmentdataPN    = segmentinfo[1]
    segmentrate      = segmentinfo[0]   # Should be 48000 for the WAVE project 
    segmentformat    = segmentinfo[3]   # Should be 16bit PCM
    hashresult       = segmentinfo[4]
    segmentduration   = math.floor(len(segmentdataPN)//(segmentrate))
    if segmentformat != paInt16:
        print("Error, PN file %s format is not Int16", subjectfile) 
        sys.exit(-1)   
    if hashresult != True:
        print("Error, PN file %s appears corrupted (failed hash check)", subjectfile) 
        sys.exit(-1)   

    # Open the subject file (recorded audio played out by the device under test) and extract the L channel
    subjectinfo    = opendatafile(subjectfile, verifyhash=False)
    rawfiledata    = subjectinfo[1]
    samplerate     = subjectinfo[0]
    sampleformat   = subjectinfo[3] # Can also fetch channels as subjectinfo[2] if needed
    
    # Trim off any leading (useless) audio prior to the watermarked portion
    #segmentdatafront = segmentdataPN[0:(2*samplerate)].copy()   # Use the first two seconds of PN data
    #segmentdatarear  = segmentdataPN[len(segmentdataPN)-(2*samplerate):].copy()   # Use the last two seconds of PN data
    filedata = trimaudio(rawfiledata, segmentdataPN, cyclicpad, NONE_TYPE)
    
    duration   = math.floor(len(filedata)/samplerate)
    if duration != segmentduration:
        print("Error, the two files have different durations, should be 60 seconds") 
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
        if neighborstart < 0:
            neighborstart = 0
        neighborend   = (idx*observationperiod)+(neighborhood//2)
        if neighborend > len(filedata):
            neighborend = len(filedata)

        subjectdata = filedata[neighborstart:neighborend].copy()
        thissegment = segmentdataPN[(idx*observationperiod):((idx+1)*observationperiod)].copy()
        resultarr[idx] = gettimefromsegment(subjectdata, thissegment, cyclicpad, NONE_TYPE) + neighborstart
        
    return resultarr

