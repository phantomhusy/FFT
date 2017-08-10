import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.fftpack import ifft,fft,fftfreq,fftshift
from scipy.interpolate import interp1d
from joblib import Parallel, delayed

FREQUENCESTEP = 0.01    # smaller, wider time range
FREQUENCEMAX = 500        # Larger, better time resolution


# load data spectrum(wavelength)
def Loaddata(filename,PLOT=None):
    wavelength,spectrum = np.loadtxt(filename, unpack=True)
    spectrum = spectrum/spectrum.max()
    if PLOT is None:
        print("Not Plot DATA.")
    else:
        plt.plot(wavelength,spectrum,"r",lw=0.8,label="spectrum")
        plt.xlabel("Wavelength [nm]",fontsize=12)
        plt.ylabel("Intensity ",fontsize=12)
        plt.grid(True)
#        plt.savefig("plot/Inten_wavelength.png",dpi=300)
    return np.c_[wavelength,spectrum]


# Covert data to spectrum(frequency)
def Interplatedata(data,PLOT=None):
    
    wavelength = data[:,0]
    spectrum   = data[:,1]
   
    #define frequency and interpolate
    frequency = 300/wavelength #*np.pi
    #frequency in 1/fs
    
    freq = frequency[::-1]
    spec = spectrum[::-1]
    # adding a point at FREQUENCEMAX
    freq = np.append(freq,FREQUENCEMAX)
    spec = np.append(spec,0)
    # adding a point at FREQUENCE=0
    freq = np.insert(freq,0,0)
    spec = np.insert(spec,0,0)
    
    fs = interp1d(freq,spec)
    freq_linear = np.arange(freq.min(),freq.max(),FREQUENCESTEP)
    spec_linear = fs(freq_linear)
    selection = spec_linear<0.02
    spec_linear[selection] = 0
    
    if PLOT is None:
        print("Not Plot DATA.")
    else:
        plt.plot(freq_linear,spec_linear.real,color='g',alpha=1)
        plt.xlabel("Frequency [1/fs]",fontsize=12)
        plt.ylabel("Intensity ",fontsize=12)
        #plt.xlim([0,120])
        plt.grid(True)
#        plt.savefig("plot/Inten_Freq.png",dpi=300)
    return np.c_[freq_linear,spec_linear]

def FFT2TimeDomain(data):
    freq = data[:,0]
    spec = data[:,1]
    E = np.absolute(spec**0.5)*np.exp(0j)
    E = fft(E)
    E = fftshift(E)/np.absolute(E).max()
    E = np.append(E,E[0])
    t = fftfreq(freq.size, FREQUENCESTEP) 
    t = fftshift(t)
    t = np.append(t,t[0])
    
    return t,E

def PlotPulse1(t,E1,TIMERANGE):
    selection = np.abs(t)<TIMERANGE  
    plt.plot(t[selection],np.absolute(E1[selection]**2),lw=2,alpha=0.5,label='Intensity')
    plt.plot(t[selection],E1[selection].real,label='E-Field')
    plt.grid(True)
    plt.legend()
    plt.xlabel("Time [fs]",fontsize=12)
    plt.ylim([-1.2,1.2])
#    plt.savefig("plot/Pulse1.png",dpi=300)
    
def PlotPulse2(t,E1,E2,TIMERANGE):
    selection = np.abs(t)<TIMERANGE  
    plt.plot(t[selection],np.absolute(E1[selection]**2),"b",lw=2,alpha=0.5,label='IR Intensity')
    plt.plot(t[selection],E1[selection].real,"g",label='IR E-Field')
    plt.plot(t[selection],np.absolute(E2[selection]**2),"b:",lw=2,alpha=0.5,label='UV Intensity')
    plt.plot(t[selection],E2[selection].real,"g:",label='UV E-Field')
    #plt.grid(True)
    #plt.legend()
    plt.xlabel("Time [fs]",fontsize=12)
    plt.ylim([-1.2,1.2])
#    plt.savefig("plot/Pulse2.png",dpi=300)

    
def PlotSignalFile(signalfile,cut=[0,120,0,350]):
    final = np.rot90(np.genfromtxt(signalfile))
    dx = 120./final.shape[1]
    dy = 350./final.shape[0]
    final =  final[500-int(cut[3]/dy):500-int(cut[2]/dy),
                   int(cut[0]/dx):int(cut[1]/dx)]
    plt.imshow(final,
               extent=[cut[0],cut[1],cut[2],cut[3]],aspect="auto",
               vmax=300, vmin =50, cmap="viridis")
    cbar = plt.colorbar()
    cbar.set_label("Signal Intensity")
    plt.xlabel("time [fs]")
    plt.ylabel("momentum [eV]")
    plt.grid(True, color="w")
#    plt.savefig("plot/final.png",dpi=300)
    

    
#############################################################
# Phase-retrieval algorithm for the characterization of 
# broadband single attosecond pulses
#############################################################
def IR_RecoE(time,ir_para):
    ir_amp  = ir_para[0]
    ir_phase= ir_para[1]
    # Equation(4)
    #temp     = np.exp(1j*(ir_omega*time+ir_phase))
    temp     = np.exp(1j*ir_phase)
    temp     = temp.real
    temp     = ir_amp * temp
    return temp

def IR_RecoA(time,ir_para):
    ir_recoE = IR_RecoE(time,ir_para)
    dt       = time[1]-time[0]
    # Equation(1-2)
    temp     = [np.sum(ir_recoE[:n]) for n in range(time.size)]
    temp     = -dt*np.array(temp)
    return temp

def IR_RecoPHI(p,time,ir_para):
    ir_recoA = IR_RecoA(time,ir_para)
    dt       = time[1]-time[0]
    # Equation(2)
    temp     = [np.sum(p*ir_recoA[n:]+0.5*ir_recoA[n:]**2) for n in range(time.size)]
    temp     = dt*np.array(temp)
    return temp

def UV_RecoE (time,uv_para):
    uv_amp   = uv_para[0]
    uv_phase = uv_para[1]
    # Equation(4)
    #temp     = np.exp(1j*(uv_omega*time+uv_phase))
    temp     = np.exp(1j*uv_phase)
    temp     = temp.real
    temp     = uv_amp * temp
    return temp

def Signal_RecoPhase(p,time,ir_para):
    ip = 0
    return (0.5*p**2 +ip)*time - IR_RecoPHI(p,time,ir_para)


def Signal_Reco(prange,delayrange,time,ir_para,uv_para,threading=1):
    dt         = time[1]-time[0]
    maxdelay   = -delayrange[0]
    delayrange = np.arange(-maxdelay,maxdelay,dt)

    if threading>1 :
        signal_phase= Parallel(n_jobs=threading)(delayed(Signal_RecoPhase)(p,time,ir_para) for p in prange)
    else:
        signal_phase= np.array([Signal_RecoPhase(p,time,ir_para) for p in prange])
    signal_amp      = UV_RecoE(time,uv_para)
    signal_amp      = np.pad(signal_amp,(int(maxdelay/dt),int(maxdelay/dt)+1),'constant', constant_values=(0, 0))
    signal_ampshift = np.array([signal_amp[  int((maxdelay-tau)/dt) : int((maxdelay+100-tau)/dt)+1] for tau in delayrange])
    
    realpart        = np.dot(np.cos(signal_phase),signal_ampshift.T) * dt
    imagpart        = np.dot(np.sin(signal_phase),signal_ampshift.T) * dt
    
    
    return realpart**2 + imagpart**2


# This function is too slow, not being used.
def SignalPoint_Reco (p,tau,time,ir_para,uv_para):
    dt     = 0.002 #time[1]-time[0]
    ip     = 21.5 #(eV)ionization energy of Neon
    ilimit = 50
    icut   = np.abs(time)<ilimit
    signal_phase = (0.5*p**2+ip)*time - IR_RecoPHI(p,time,ir_para)
    signal_phase = signal_phase[icut]
    signal_amp   = UV_RecoE(time,uv_para)
    if tau >=0:
        shiftright = abs(int(tau/dt))
        signal_ampshift = np.concatenate([np.zeros(shiftright),signal_amp]
                                        )[:signal_amp.size]
    else:
        shiftleft  = abs(int(tau/dt))
        signal_ampshift = np.concatenate([signal_amp,np.zeros(shiftleft)]
                                        )[-signal_amp.size:]
    signal_ampshift   = signal_ampshift[icut]
    temp = signal_ampshift * np.exp(1j*signal_phase)
    temp = temp * dt
    return (np.sum(temp.real)**2 + np.sum(temp.imag)**2)