import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os,argparse

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
width=0.75
color='black'
fontsize=28
ticksize=22
figsize=(10,10)

def main(datafile):

    df = pd.read_csv(datafile)

    # print(df.columns)

    errs = 1-np.array(df[df['method']=='hybrid']['relative_error'])
    cerrs = 1-np.array(df[df['method']=='classical']['relative_error'])

    Ds = np.array(df[df['method']=='hybrid']['circuit_depth'])
    chis = np.array(df[df['method']=='hybrid']['chi'])
    fhparas = np.array(df[df['method']=='hybrid']['params_original'])

    ND = len(np.unique(Ds))
    Nchi = len(np.unique(chis))

    fcparas = np.array(df[df['method']=='classical']['params_original'])

    hparas = np.array(df[df['method']=='hybrid']['total_params'])/fhparas
    cparas = np.array(df[df['method']=='classical']['total_params'])/fcparas

    errs = errs.reshape((ND, Nchi))
    Ds = Ds.reshape(errs.shape)
    chis = chis.reshape(errs.shape)
    hparas = hparas.reshape(errs.shape)

    # print(errs)
    # print(Ds)
    # print(chis)

    # inpath = os.path.dirname(os.path.abspath(datafile))
    outfile = '.'.join(datafile.split('.')[:-1])+'_accuracy_over_depth_chi.png'

    # fig, ax = plt.subplots(Nchi, 1, figsize=(figsize[0], figsize[1]*Nchi))
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()

    colors = plt.cm.viridis(np.linspace(0,1,Nchi))
    for i in range(Nchi):
        ax.plot(Ds[:,0], errs[:,i], color=colors[i], label=r'$\chi=$'+str(chis[0,i]))

    ax.set_yscale('log')
    ax.set_xscale('log')

    plt.savefig(outfile, bbox_inches='tight')


    outfile = '.'.join(datafile.split('.')[:-1])+'_paras.png'

    fig = plt.figure(figsize=figsize)
    ax = fig.gca()

    ax.plot(cparas, cerrs, color='black', alpha=0.7, lw=0.2, ls='--')

    for i in range(Nchi):
        ax.scatter(hparas[1:,i], errs[1:,i], color=colors[i], alpha=0.7)
        ax.scatter(cparas[i], cerrs[i], color=colors[i], alpha=0.7, marker='x', s=50)

    ax.set_yscale('log')
    ax.set_xscale('log')

    plt.savefig(outfile, bbox_inches='tight')

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="",
    )

    parser.add_argument("infile")

    args = parser.parse_args()
    raise SystemExit(main(args.infile))
