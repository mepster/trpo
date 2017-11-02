#! /usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pickle

if 0:
    k1=0.5
    k2=0.5
    k3=0.5
    x0=0.0
    sample=400
    x = np.arange(sample)/100
    x2 = x/(1.0+np.exp(-1.0*k1*(x-x0)))

    l_targ = -0.1+0.4*np.sin(np.pi*x2)
    r_targ = -0.1+0.4*np.sin(np.pi*x2+np.pi)
    plt.plot(x, l_targ*np.exp(-k3*x))
    plt.plot(x, r_targ*np.exp(-k3*x))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def plot_one(ax_x, ax_y, actname, targname, act, targ):
    actx = [p[0] for p in act]
    acty = [p[1] for p in act]
    targx = [p[0] for p in targ]
    targy = [p[1] for p in targ]

    ax_x.plot(actx, color='r')
    ax_x.plot(targx, color='b')
    ax_x.set_ylabel(actname+"_x", rotation=0, labelpad=30)
    ax_y.plot(acty, color='r')
    ax_y.plot(targy, color='b')
    ax_y.set_ylabel(actname+"_y", rotation=0, labelpad=30)

def plot_all(trace):
    #plot_pos("head", trace)

    serieslist = ["head", "talus_l", "talus_r"] 
    num = 2*len(serieslist)
    f, ax = plt.subplots(num, sharex=True, sharey=True)

    i = 0
    for actname in serieslist:
        targname = actname+"_targ"
        act = trace[actname]
        targ = trace[targname]
        plot_one(ax[i], ax[i+1], actname, targname, act, targ)
        i = i+2
    
    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    f.subplots_adjust(hspace=0, left=0.25)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    print(plt.margins())
    
    plt.show()

def plot_one_test(ax_x, ax_y, targname, targ):
    targx = [p[0] for p in targ]
    targy = [p[1] for p in targ]

    ax_x.plot(targx, color='b')
    ax_x.set_ylabel(targname+"_x", rotation=0, labelpad=30)
    ax_y.plot(targy, color='b')
    ax_y.set_ylabel(targname+"_y", rotation=0, labelpad=30)

BLANK = None

def targs(x):
    x0=0.0
    k2=2
    x2=x * (2/(1.0+np.exp(-k2*(x-x0)))-1.0)

    k3=10.0
    x3=2/(1.0+np.exp(-k3*(x-x0)))-1.0

    head_targ = np.array([0.5*x3, BLANK])
    talus_l_targ = np.array([0.4*np.sin(np.pi*x2+np.pi),       BLANK])
    talus_r_targ = np.array([0.4*np.sin(np.pi*x2), BLANK])

    return (head_targ, talus_l_targ, talus_r_targ)

def plot_test():

    head_targ=[]
    talus_l_targ=[]
    talus_r_targ=[]
    trace = {'head_targ': [], 'talus_l_targ':[], 'talus_r_targ':[]}
    for x in range(3000):
        (h, tl, tr) = targs(x*1e-3)
        trace['head_targ'].append(h)
        trace['talus_l_targ'].append(tl)
        trace['talus_r_targ'].append(tr)
    print(trace['head_targ'])
    
    serieslist = ["head", "talus_l", "talus_r"] 
    num = len(serieslist)
    f, ax = plt.subplots(2*num, sharex=True, sharey=True)

    i = 0
    for actname in serieslist:
        targname = actname+"_targ"
        targ = trace[targname]
        plot_one_test(ax[i], ax[i+1], targname, targ)
        i = i+2
    
    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    f.subplots_adjust(hspace=0, left=0.25)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    print(plt.margins())
    
    plt.show()

if __name__ == "__main__":
    with open("trace", 'rb') as f:
        if 0:
            plot_test()
        else:
            trace = pickle.load(f)
            plot_all(trace)

    
