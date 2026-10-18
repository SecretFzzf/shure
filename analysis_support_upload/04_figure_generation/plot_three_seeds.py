"""Generate manuscript Figures 1-4 from the transmission analysis tables."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
 here=Path(__file__).resolve().parent
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=here.parent/'02_transmission_analysis');p.add_argument('--out',type=Path,default=here);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
 names={'mean':'Mean','spline':'Spline','knn':'KNN','brits':'BRITS','saits':'SAITS'};colors=['#555555','#0072B2','#CC79A7','#009E73','#D55E00'];models=['LSTM','DLinear','PatchTST']
 c=pd.read_csv(a.data/'figure1_coordinates.csv');fig,axs=plt.subplots(1,3,figsize=(9,3.6),sharey=True)
 for ax,m in zip(axs,models):
  for (method,name),color in zip(names.items(),colors):
   x=c[(c.model==m)&(c.impute_method==method)].sort_values('missing_rate');ax.plot(x.missing_rate*100,x.fe_mae,label=name,color=color,marker='o',ms=4,ls='--' if method=='mean' else '-')
  ax.set(title=m,xlabel='Missing rate (%)',xticks=[10,30,50,70]);ax.grid(alpha=.15)
 axs[0].set_ylabel('Median forecasting MAE');fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=5,frameon=False);fig.tight_layout(rect=(0,0,1,.88));fig.savefig(a.out/'fig1.png');plt.close(fig)
 t=pd.read_csv(a.data/'main_contrasts_2304.csv');fig,axs=plt.subplots(1,3,figsize=(9,3.8),sharex=True,sharey=True)
 for ax,m in zip(axs,models):
  x=t[t.model==m];valid=x.valid.astype(bool);pure=x.pure.astype(bool)
  for mask,col,label in [(~valid,'#bdbdbd','Ratio excluded'),(valid&~pure,'#0072B2','Other valid'),(pure,'#D55E00','Pure inversion')]:ax.scatter(x.loc[mask,'delta_ie'],x.loc[mask,'delta_fe'],s=12,alpha=.55,color=col,label=label,rasterized=True)
  ax.axhline(0,color='black',lw=.6);ax.axvline(0,color='black',lw=.6);ax.axvspan(-.01,.01,color='grey',alpha=.15);ax.set(title=m,xlabel=r'$\Delta\mathrm{IE}$')
 axs[0].set_ylabel(r'$\Delta\overline{\mathrm{FE}}$');fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False);fig.tight_layout(rect=(0,0,1,.86));fig.savefig(a.out/'fig2.png');plt.close(fig)
 s=pd.read_csv(a.data/'threshold_sensitivity.csv');s=s[s.metric=='MAE'];fig,ax=plt.subplots(figsize=(6.7,3.6));ax.plot(s.epsilon,s.pure_pct,'o-',label='Pure inversion',color='#D55E00');ax.plot(s.epsilon,s.global_pct,'s-',label='Global inversion',color='#0072B2');ax.set(xlabel=r'Ratio threshold $\epsilon$',ylabel='Inversion (%)');ax.legend(frameon=False);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(a.out/'fig3.png');plt.close(fig)
 h=pd.read_csv(a.data/'heatmap.csv').pivot(index='model',columns='impute_method',values='pure_pct').reindex(index=models,columns=['spline','knn','brits','saits']);fig,ax=plt.subplots(figsize=(6.7,3.5));im=ax.imshow(h,cmap='YlOrRd',vmin=0,vmax=max(40,h.max().max()));ax.set_xticks(range(4),['Spline','KNN','BRITS','SAITS']);ax.set_yticks(range(3),models)
 for i in range(3):
  for j in range(4):ax.text(j,i,f'{h.iloc[i,j]:.1f}%',ha='center',va='center',color='white' if h.iloc[i,j]>30 else 'black')
 fig.colorbar(im,ax=ax,label='Pure inversion (%)');fig.tight_layout();fig.savefig(a.out/'fig4.png');plt.close(fig)
if __name__=='__main__':main()
