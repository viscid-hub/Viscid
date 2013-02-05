#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import pylab as pl
import vggcm
import fggcm
import subprocess as sub
import csv

def my_boxplot(what, color='blue', **args):
    b = pl.boxplot(what, **args)
    pl.setp(b['boxes'],color=color)
    pl.setp(b['whiskers'],color=color)
    pl.setp(b['caps'],color=color)
    pl.setp(b['medians'],color=color)
    return b

if __name__=='__main__':
    parser = vggcm.optargs()
    parser.add_option('-l', '--lims', dest='lims', action='callback',
                      callback=vggcm.optlist_float, type='string',
                      default=[-20, -8], help='Limits of the line slice (default: -20,-8)')
    parser.add_option('-n', '--noplot', dest='noplot', action='store_true',
                      help='do not create any plots.')
    (opt, args) = parser.parse_args()
    global verb
    verb = 1 + opt.v - opt.q

    args = fggcm.expand_fnames(args)

    if len(args) == 0:
        raise ValueError("must supply at least one file name argument")
    collections = vggcm.collect_files(args, collectby='run')

    # read in resistivities from my csv file
    resis = [None for c in collections]
    with open(fggcm.get_runs_csv_fname(), 'rb') as f:
        j=0
        reader = csv.reader(f)
        for line in reader:
            for i, c in enumerate(collections):
                if line[0] == c[0].finfo['run']:
                    resis[i] = float(line[1])
                    j += 1
                    break
            if j == len(resis):
                break
        if j != len(resis):
            raise LookupError('Could not find one or more of the runs in runs.csv')

    # read in the data
    # these vars are indexed by x[run][timeslice][x position]
    x = [None for c in collections]
    y = [None for c in collections]
    z = [None for c in collections]
    time = [None for c in collections]
    #resis = [None for c in collections]
    bz = [None for c in collections]
    vx = [None for c in collections]
    rr = [None for c in collections]
    xjy = [None for c in collections]

    xjysheetwidth = [[None for f in c] for c in collections]
    xjyleftpos = [[None for f in c] for c in collections]
    xjyrightpos = [[None for f in c] for c in collections]
    xjyleft = [[None for f in c] for c in collections]
    xjyright = [[None for f in c] for c in collections]
    xjyextremeind = [[None for f in c] for c in collections]
    xjyextremepos = [[None for f in c] for c in collections]
    xjyextreme = [[None for f in c] for c in collections]
    #A = [[None for f in c] for c in collections]
    #B = [[None for f in c] for c in collections]
    #C = [[None for f in c] for c in collections]

    for i, c in enumerate(collections):
        # get the line from the first time slice in each run
        time[i] = [f['time'] for f in c]
        #resis[i] = c[0]['resis'][0,0]
        cx, cy, cz = c[0]['coords']
        #yind = np.argmin(np.abs(cy))
        zind = np.argmin(np.abs(cz))
        #print("zind: ", zind)
        cxarray = np.array(cx)
        xindleft = np.argmin(np.abs(cxarray-opt.lims[0]))
        xindright = np.argmin(np.abs(cxarray-opt.lims[1]))
        stride = -1 if xindleft > xindright else 1
        #print(np.shape(cx), np.shape(c[0]['b'][0]))
        x[i] = [cx[xindleft:xindright:stride] for f in c]
        y[i] = [[cy]*len(x[i]) for f in c]
        z[i] = [[cz[zind]]*len(x[i]) for f in c]
        bz[i] = [f['b'][2][zind,xindleft:xindright:stride] for f in c]
        vx[i] = [f['v'][0][zind,xindleft:xindright:stride] for f in c]
        xjy[i] = [f['xj'][1][zind,xindleft:xindright:stride] for f in c]
        rr[i] = [f['rr'][zind,xindleft:xindright:stride] for f in c]

        # now find current sheet width and x and stagnation points and stuff
        for j, f in enumerate(c):
            _x = x[i][j]
            _vx = vx[i][j]
            _bz = bz[i][j]
            _xjy = xjy[i][j]
            _rr = rr[i][j]
            _resis = resis[i]
            '''pl.plot(_x, _bz)
            pl.show()
            pl.clf()'''

            _dvx_dx = np.zeros_like(_x)
            _dvx_dx[1:-1] = (_vx[2:] - _vx[0:-2]) / (_x[2:] - _x[0:-2])
            _dbz_dx = np.zeros_like(x[i][j])
            _dbz_dx[1:-1] = (_bz[2:] - _bz[0:-2]) / (_x[2:] - _x[0:-2])

            _xjymax = np.max(_xjy)
            _xjymin = np.min(_xjy)
            if np.abs(_xjymax) > np.abs(_xjymin):
                _xjyextremeind = np.argmax(_xjy)
                _peaktype = 'max'
            else:
                _xjyextremeind = np.argmin(_xjy)
                _peaktype = 'min'

            _xjyrightind = _xjyextremeind
            _xjyleftind = _xjyextremeind
            # fit a polynomial to the 3 peak points to get a better estimate of the max
            _a, _b, _c = np.polyfit(_x[_xjyextremeind-1:_xjyextremeind+2],
                                 _xjy[_xjyextremeind-1:_xjyextremeind+2], 2)
            _xjyextremepos = -0.5 * _b / _a
            _xjyextreme = _a*_xjyextremepos**2 + _b*_xjyextremepos +  _c

            try:
                while(np.abs(_xjy[_xjyrightind]) >= 0.5*np.abs(_xjyextreme)):
                    _xjyrightind += 1
                while(np.abs(_xjy[_xjyleftind]) >= 0.5*np.abs(_xjyextreme)):
                    _xjyleftind -= 1
                if _peaktype == 'max':
                    _xjyleftpos = np.interp(0.5*_xjyextreme, _xjy[_xjyleftind:_xjyleftind+2],
                                                               _x[_xjyleftind:_xjyleftind+2])
                    _xjyrightpos = np.interp(0.5*_xjyextreme, _xjy[_xjyrightind:_xjyrightind-2:-1],
                                                                _x[_xjyrightind:_xjyrightind-2:-1])
                elif _peaktype == 'min':
                    _xjyleftpos = np.interp(0.5*_xjyextreme, _xjy[_xjyleftind+1:_xjyleftind-1:-1],
                                                               _x[_xjyleftind+1:_xjyleftind-1:-1])
                    _xjyrightpos = np.interp(0.5*_xjyextreme, _xjy[_xjyrightind-1:_xjyrightind+1],
                                                                _x[_xjyrightind-1:_xjyrightind+1])
            except IndexError:
                raise IndexError("Could not find half max before "
                                 "end of x range in file {0}".format(f.fname))

            # these two should just be 0.5 * xjyextreme
            _xjyleft = np.interp(_xjyleftpos, _x, _xjy)
            _xjyright = np.interp(_xjyrightpos, _x, _xjy)
            #print(_xjyleftind, _xjyextremeind, _xjyrightind)
            #print(_xjyleftpos, _x[_xjyleftind], _x[_xjyrightind], _xjyrightpos)
            #print(_xjyleft, _xjyextreme, _xjyright)

            # now find the x-point and stagnation point
            _xpoint_ind = len(_x)-1
            _stagpt_ind = len(_x)-1
            #print(_x, _bz)
            try:
                #print("looking for x-point")
                while _bz[_xpoint_ind] > 0 and _xpoint_ind >= 0:
                    #print("XP -")
                    _xpoint_ind -= 1
                    #print(_xpoint_ind, _x[_xpoint_ind], _bz[_xpoint_ind])
                    if _xpoint_ind == 0: raise IndexError
                #print("looking for stagnation point")
                while _vx[_stagpt_ind] < 0 and _stagpt_ind >= 0:
                    _stagpt_ind -= 1
                    #print(_stagpt_ind, _x[_xpoint_ind], _vx[_stagpt_ind])
                    if _stagpt_ind == 0: raise IndexError
            except IndexError:
                raise IndexError("Could not find stagnation or x points "
                                "in file {0}".format(f.fname))
            _stagpt = _x[_stagpt_ind] - (_vx[_stagpt_ind] / _dvx_dx[_stagpt_ind])
            _xpoint = _x[_xpoint_ind] - (_vx[_xpoint_ind] / _dbz_dx[_xpoint_ind])

            ## Make a plot of the ohm's law terms for each file
            if not opt.noplot:
                pl.subplot(2,1,1)
                #print(_x, _resis, _xjy)
                p1 = pl.plot(_x, _resis*_xjy, 'r', linewidth=2)
                pl.title(r"Terms in Ohm's Law")
                #pl.ylabel(r'$J_y[\mu A/m^2]')
                #print([_xjyextremepos]*2, [0, _resis*_xjyextreme])
                pl.plot([_xjyextremepos]*2, [0, _resis*_xjyextreme], 'k-', linewidth=1)
                pl.plot([_xjyleftpos]*2, [0, _resis*_xjyleft], 'k-', linewidth=1)
                pl.plot([_xjyrightpos]*2, [0, _resis*_xjyright], 'k-', linewidth=1)
                p2 = pl.plot(_x, _vx*_bz, 'b', linewidth=2)
                p3 = pl.plot(_x, _vx*_bz + _resis * _xjy, 'k', linewidth=2)
                #pl.ylim(-600, np.max(_vx*_bz))
                #pl.xlim(np.min(_x), np.max(_x))
                pl.legend([p1, p2, p3], [r'$\eta J_y$',r'$(v \times B)_y$',r'$(v \times B)_y +\eta J_y$'],loc=6)

                pl.subplot(2,1,2)
                pl.title('Component Nulls and Asymmetries')
                p4 = pl.plot(_x, _bz, 'k', linewidth=2)
                pl.plot(_x, np.zeros_like(_x), 'k--')
                #print(_xpoint, _stagpt)
                p7 = pl.plot([_xpoint]*2, [-100, 100], 'm--', linewidth=2)
                p8 = pl.plot([_stagpt]*2, [-100, 100], 'c--', linewidth=2)
                #plt.plot([xleft,xleft],[0, bzleft[i,k] ],'k',linewidth=3)
                #plt.plot([xright,xright],[0, bzright[i,k] ],'k',linewidth=3)

                p5 = pl.plot(_x, _rr, 'b', linewidth=2)
                p6 = pl.plot(_x, _vx, 'r', linewidth=2)
                pl.legend([p4,p5,p6,p7,p8],[r'$B_z[nT]$',r'$n/cc$',r'$V_x[km/s]$','x-pnt.','stag. pnt.'],loc=6)
                pl.xlim(np.min(_x), np.max(_x))
                pl.ylim(-70,100)
                pl.xlabel(r'$X [R_e ]$')

                #pl.show()
                pl.savefig("img{0:06d}.png".format(j+1))
                pl.clf()
            xjysheetwidth[i][j] = _xjyrightpos - _xjyleftpos
            xjyextreme[i][j] = _xjyextreme
            '''xjysheetwidth[i][j] = _xjyrightpos - _xjyleftpos
            xjyleftpos[i][j] = _xjyleftpos
            xjyrightpos[i][j] = _xjyrightpos
            xjyleft[i][j] = _xjyleft
            xjyright[i][j] = _xjyright
            xjyextremepos[i][j] = _xjyextremepos
            xjyextremeind[i][j] = _xjyextremeind'''
            ##A[i][j] = _a
            #B[i][j] = _b
            #C[i][j] = _c

    if verb:
        for c, eta, widths in zip(collections, resis, xjysheetwidth):
            print("run: {0}\teta: {1:.2e}\twidth: {2:.3f} +/- {3:.3f}".format(c[0].finfo['run'], eta, np.average(widths), np.std(widths)))

    if not opt.noplot:
        runname = fggcm.name_parse(c[0].fname, True)[1]
        sub.Popen("ffmpeg -r 5 -qscale 1 -i img%06d.png nulls_and_asymmetries_{0}.mp4".format(runname),
                  shell=True).communicate()
        sub.Popen("rm img*.png".format(), shell=True).communicate()

        b1 = my_boxplot(xjysheetwidth, positions=resis, widths=np.array(resis)/4, sym='', color='blue')
        linedummy = np.array([.1, 1e5])
        pl.plot(linedummy, .001*linedummy**0.5, 'r--')
        pl.xscale('log')
        pl.yscale('log')
        pl.xlim(1e3, 1e5)
        #pl.ylim(.1, 2)
        pl.ylabel(r'FWHM of Subsolar $J_y$')
        pl.xlabel(r'$\eta$')
        pl.title('Resistive Scaling of Current Sheet Width')
        pl.savefig('thickness_scaling.png',dpi=100)
        pl.clf()

        boxit = [np.abs(np.array(jy))*r for jy, r in zip(xjyextreme, resis)]
        b2 = my_boxplot(boxit, positions=resis, widths=np.array(resis)/4, sym='', color='blue')
        linedummy = np.array([.1, 1e5])
        pl.plot(linedummy, 100+.001*linedummy**0.5, 'r--')
        pl.ylabel(r'Subsolar Peak $\eta J$')
        pl.xlabel(r'$\eta$')
        pl.xscale('log')
        pl.yscale('log')
        pl.xlim(1e3,1e5)
        #pl.ylim(10,5e3)
        pl.title(r'Resistive Scaling of Subsolar peak $\eta J$')
        pl.savefig('eta_j_scaling.png',dpi=100)


    #pl.show()


##
## EOF
##

