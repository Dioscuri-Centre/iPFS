#%% 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl 
import pandas as pd 
import os
import pandas as pd
import plotly.graph_objects as go 

glasbey = ['#d60000', '#8c3bff', '#018700', '#00acc6', '#97ff00', '#ff7ed1', '#6b004f', '#ffa52f', '#573b00', '#005659', '#0000dd', '#00fdcf', '#a17569', '#bcb6ff', '#95b577', '#bf03b8', '#645474', '#790000', '#0774d8', '#fdf490', '#004b00', '#8e7900', '#ff7266', '#edb8b8', '#5d7e66', '#9ae4ff', '#eb0077', '#a57bb8', '#5900a3', '#03c600', '#9e4b00', '#9c3b4f', '#cac300', '#708297', '#00af89', '#8287ff', '#5d363b', '#380000', '#fdbfff', '#bde6bf', '#db6d01', '#93b8b5', '#e452ff', '#2f5282', '#c36690', '#54621f', '#c49e72', '#038287', '#69e680', '#802690', '#6db3ff', '#4d33ff', '#85a301', '#fd03ca', '#c1a5c4', '#c45646', '#75573d', '#016742', '#00d6d4', '#dadfff', '#f9ff00', '#6967af', '#c39700', '#e1cd9c', '#da95ff', '#ba03fd', '#915282', '#a00072', '#569a54', '#d38c8e', '#364426', '#97a5c3', '#8e8c5e', '#ff4600', '#c8fff9', '#ae6dff', '#6ecfa7', '#bfff8c', '#8c54b1', '#773618', '#ffa079', '#a8001f', '#ff1c44', '#5e1123', '#679793', '#ff5e93', '#4b6774', '#5291cc', '#aa7031', '#01cffd', '#00c36b', '#60345d', '#90d42f', '#bfd47c', '#5044a1', '#4d230c', '#7c5900', '#ffcd44', '#8201cf', '#4dfdff', '#89003d', '#7b525b', '#00749c', '#aa8297', '#80708e', '#6264fd', '#c33489', '#cd2846', '#ff9ab5', '#c35dba', '#216701', '#008e64', '#628023', '#8987bf', '#97ddd4', '#cd7e57', '#d1b65b', '#60006e', '#995444', '#afc6db', '#f2ffd1', '#00eb01', '#cd85bc', '#4400c4', '#799c7e', '#727046', '#93ffba', '#0054c1', '#ac93eb', '#3fa316', '#5e3a80', '#004b33', '#7cb8d3', '#972a00', '#386e64', '#b8005b', '#ff803d', '#ffd1e8', '#802f59', '#213400', '#a15d6e', '#4fb5af', '#9e9e46', '#337c3d', '#c14100', '#c6e83d', '#6b05e8', '#75bc4f', '#a5c4a8', '#da546e', '#d88e38', '#fb7cff', '#4b6449', '#d6c3eb', '#792d36', '#4b8ea5', '#4687ff', '#a300c3', '#e9a3d4', '#ffbc77', '#464800', '#a1c6ff', '#90a1e9', '#4f6993', '#e65db1', '#9e90af', '#57502a', '#af5dd4', '#856de1', '#c16e72', '#e400e2', '#b8b68a', '#382d00', '#e27ea3', '#ac3b2f', '#a8ba4b', '#69b582', '#93d190', '#af8c46', '#075e77', '#009789', '#590f01', '#5b7c80', '#2f5726', '#e4643b', '#5e3f28', '#7249bc', '#4b526b', '#c879dd', '#9c3190', '#c8e6f2', '#05aaeb', '#a76b9a', '#e6af00', '#60ff62', '#f2dd00', '#774401', '#602441', '#677eca', '#799eaf', '#0ce8a1', '#9cf7db', '#830075', '#8e6d49', '#e2412f', '#b8496b', '#794985', '#ffcfb5', '#4b5dc6', '#e2b391', '#ff4bed', '#d6efa5', '#bf6026', '#d6a3b5', '#bc7c00', '#876eb1', '#ff2fa1', '#ffe8af', '#33574b', '#b88c79', '#6b8752', '#bc93d1', '#1ae6fd', '#a13b72', '#a350a8', '#6d0097', '#89647b', '#59578a', '#f98e8a', '#e6d67c', '#706b01', '#1859ff', '#1626ff', '#00d856', '#f7a1fd', '#79953b', '#b1a7d4', '#7ecfdd', '#00caaf', '#79463b', '#daffe6', '#db05b1', '#f2ddff', '#a3e46e', '#891323', '#666782', '#e8fd70', '#d8aae8', '#dfbad4', '#fd5269', '#75ae9a', '#9733df', '#e4727e', '#8c5926', '#774669', '#2f3da8']


cm = 1/2.54  # centimeters in inches
# 
def redondo(img,debug=False):
    ''' 
    Reimplement redondo ( Laplace) with scipy, this matches MM implementation, which follows: 
    >>> def redondo2(img):
    >>>     # copy of the implementation of MM 
    >>>     h,w = img.shape
    >>>     ret = np.zeros((h,w),dtype=float)
    >>>     for j in range(1,h-1):
    >>>         for i in range(1,w-1):
    >>>             p = img[j,i-1]+img[j,i+1]+img[j-1,i]+img[j+1,i]-4*img[j,i-1]
    >>>             ret[j,i] = p**2
    >>>     return ret
    >>> img = stack.get_img(0,0,0,0)[0]
    >>> redo = redondo(img)
    >>> redo2 = redondo2(img)
    >>> (redo[1:-1,1:-1] == redo2[1:-1,1:-1]).all()
    '''
    import cv2
    ddepth = cv2.CV_64F
    # Apply identity kernel
    redondo_k =np.array([[0, 1, 0],
                        [-3, 0, 1],
                        [0, 1, 0]])

    convolved = cv2.filter2D(src=img.astype(float), ddepth=-1, kernel=redondo_k,delta=0,borderType=cv2.BORDER_ISOLATED)
    return convolved**2



# Get a vectorised function that retrieves XYZ positions from metadata 

def get_xyz(x): 
    return np.vectorize(lambda X: np.array([X['map']['XPositionUm']['scalar'],
                                           X['map']['YPositionUm']['scalar'],
                                           X['map']['ZPositionUm']['scalar']],
                                         )
                        , signature='()->(n)'
                        )(x)
def get_time(x): 
    x = np.vectorize(lambda X: X['map']['ReceivedTime']['scalar'][:-5])(x)
    x = x.astype('datetime64[s]')
    return x 

def parse_data(folder):
    dat = pd.read_csv(os.path.join(folder,'logs_sharpness.txt'),
                      sep='\t',names=['t', 'idx', 'z','sh_max','time'],header=0)
    dat.time = pd.to_datetime(dat.time)
    dat_raw  = dat.copy() 
    
    # Make a dict 
    dat_new  = {} 
    for i in dat.idx.unique(): 
#        dat_new[i] = dat[dat['idx'] == i ].dropna().drop(columns='idx').set_index('t')
        # alternative 
        
        df = dat[dat['idx'] == i ].dropna().reset_index(drop=True).drop(columns='idx')
        df['delta_time'] = ( df.time - dat.time.min())/np.timedelta64(1, "h")
        #df.set_index('time',inplace=True)
        dat_new[i] = df 
    dat = dat_new
    ids = dat.keys()
    dat = pd.concat(dat.values(),keys=dat.keys(),axis=1,names=['dot index'])

    return dat

def plot_exp(dat,qty,style='mpl',fig=None,lgd=True):
    # Alternate solution using multiindex Dataframes (which sucks)
    # dat = pd.concat ( [dat[dat.idx == i] for i in dat.idx.unique() ] ,keys = dat['idx'].unique(),names=['idx'],axis=1 )    
    # # list of unique labels  
    # for i in dat.columns.levels[0] : 
    #     times = dat[i].dropna().time
    #     # we need iloc otherwise 
    #     dat.loc[:,(i,'delta_time')]= ( times - times.iloc[0]) /np.timedelta64(1, "m")
    # ids =  dat.columns.levels[0]
    # return dat 

    # Alternate solution but this time grouped by rows (sucks even more, I cannot make it work ... )
#     dat = pd.concat ( [dat[dat.idx == i] for i in dat.idx.unique() ] ,keys = dat['idx'].unique(),axis=0 )    
#     # list of unique labels  
#     dat = dat.set_index(['idx','t'],verify_integrity=True).sort_index()
#     dat = dat.sort_values(by='t')
#     return dat 

#     for i in dat.index.levels[0] : 
#         times = dat.loc[i].time
#         # we need iloc otherwise 
#         dat.loc[(i,slice(None)),'delta_time']= ( times - dat.time[0]) #/np.timedelta64(1, "m")
#     ids = dat.index.levels[0]
#     return dat 

    if style == 'mpl':
        # matplotlib style
        if not fig:
            fig, ax = plt.subplots()
        else:
            ax = fig.gca()
        for i in dat.columns.levels[0]:
            df = dat[i]
            ax.plot(df['delta_time'],df[qty],label=i)
        #fig.autofmt_xdate()
        
        ax.set_ylabel(qty)
        if lgd:
            ax.legend(loc='right')
            ax.legend(bbox_to_anchor=(1.05, 1.0),title= 'dot index')
        return fig
    
    elif style == 'pyplot':
        # Multidimensional index b
        layout = dict(
        #autosize=True,
        width = 1000,
        height =1000,
        xaxis_title="time (hours)",
        #yaxis_title= yaxis_title,
        legend_title="Dots",
        # font=dict(
        #     family="Courier New, monospace",
        #     size=18,
        #     color="RebeccaPurple"
        # )
        )
        fig = go.Figure(layout=layout)
        
        for i in dat.columns.levels[0]:
            df = dat[i]
            fig.add_trace(go.Scatter(x=df['delta_time'], y=df['sh_max'],mode='lines', name=f"{i}",text=f"{i}"))
        # plt.legend()
        fig.show()
        
    #return dat


# def plot_single_dot(folder,i=0):
#     dat = pd.read_csv(os.path.join(folder,'logs_sharpness_raw.txt'),
#                       sep='\t',names=['t', 'idx', 'z','sh','time'],header=0,infer_datetime_format=True)
    
#     dat['time'] = pd.to_datetime(dat['time'])
    
#     #df = dat[dat['idx'] == i]
#     #plt.plot(df['t'],df['z'],label = i)
#     #plt.legend()
    
#     plt.figure()
#     plt.title(f"Sharpness over time for dot {i}")
#     dat = dat[dat['idx'] == i]
#     # plot the raw sharpness functions over the time 
#     x = sum(  (df['idx'] == i) & (df['t'] == 0) )
#     for t in dat['t'].unique():
#         d = dat.loc[ dat['t'] == t  ]
#         plt.plot(d['z'],d['sh'],'-',label = f"time {t}")
#     plt.legend()
#     return dat


def convert_data_to_multid(dat):

     dat_new = {}
     for i in dat.idx.unique(): 
 #        dat_new[i] = dat[dat['idx'] == i ].dropna().drop(columns='idx').set_index('t')
         # alternative 
         df = dat[dat['idx'] == i ].dropna().reset_index(drop=True).drop(columns='idx')
         # if  'time' in dat.columns:
         #     df['delta_time'] = ( df.time - dat.time.min())/np.timedelta64(1, "h")
         #df.set_index('time',inplace=True)
         dat_new[i] = df 
     dat = dat_new
     #ids = dat.keys()
     dat = pd.concat(dat_new.values(),keys=dat_new.keys(),axis=1,names=['dot index'])

     return dat

def parse_data_sh(folder):
    dat = pd.read_csv(os.path.join(folder,'logs_sharpness_raw.txt'),
                      sep='\t',names=['t', 'idx', 'z','sh','time'],header=0)#,format="%Y/%M/%d %h:%m")
    if  'time' in dat.columns:
        dat['time'] = pd.to_datetime(dat['time'])
        dat['delta_time'] = ( dat.time - dat.time.min())/np.timedelta64(1, "h")

    return convert_data_to_multid(dat)
 
    
def plot_single_dot(folder,i,qty,plot=True):
    dat = pd.read_csv(os.path.join(folder,'logs_sharpness_raw.txt'),
                      sep='\t',names=['t', 'idx', 'z','sh','time'],header=0)#,format="%Y/%M/%d %h:%m")
    
    dat['time'] = pd.to_datetime(dat['time'])
    dat['delta_time'] = ( dat.time - dat.time.min())/np.timedelta64(1, "h")
    
    
    fig, ax = plt.subplots(1,2)
    df = dat[dat['idx'] == i]
    ax = ax.flat 
    ax[0].set_title(f"{qty} over time for dot {i}")
    ax[1].set_title(f" z over time for dot {i}")

    for t in df['t'].unique():
        d = df.loc[ df['t'] == t  ]
        ax[0].plot(d['delta_time'],d[qty],'-',)
        ax[1].plot(d['delta_time'],d['z'],'-',)
    #plt.legend()
    return dat,fig

def plot_single_errorbar(folder,qty):
    dat = pd.read_csv(os.path.join(folder,'logs_sharpness_raw.txt'),
                      sep='\t',names=['t', 'idx', 'z','sh','time'],header=0)
    
    dat['time'] = pd.to_datetime(dat['time'])
    dat['delta_time'] = ( dat.time - dat.time.min())/np.timedelta64(1, "h")
    fig, ax = plt.subplots(1,len(qty),figsize=(len(qty)*5,5))

    cmap = mpl.colormaps['tab20']
    cols = [ cmap(x) for x in np.linspace(0,1,20) ] 
    for j in dat['idx'].unique():
        df = dat[dat['idx'] == j]
        means = df.groupby(['t','idx',]).mean().reset_index(['t','idx'])
        std = df.groupby(['t','idx',]).std().reset_index(['t','idx'])
        shmax = df.groupby(['t','idx',]).max().reset_index(['t','idx'])
 
        for i,q in enumerate(qty):
            ax[i].set_title(f"{q} over time for dot {i}")
            ax[i].errorbar(x=means['delta_time'] ,y=means[q],yerr=std[q],color=cols[j%20])
    plt.legend()
    return dat,fig



def parse_exp_raw(folder):
    dat = pd.read_csv(os.path.join(folder,'logs_sharpness_raw.txt'),
                      sep='\t',names=['t', 'idx', 'z','sh','time'],header=0)
    
    dat['time'] = pd.to_datetime(dat['time'])
    dat['delta_time'] = ( dat.time - dat.time.min())/np.timedelta64(1, "h")
    for j in dat['idx'].unique():
        df = dat[dat['idx'] == j]

    return dat 


def plot_exp_raw(dat,qty,stderr=True,style='mpl',fig=None):
    dat = dat.copy()
    cmap = mpl.colormaps['tab20']
    cols = [ cmap(x) for x in np.linspace(0,1,20) ] 
    if style == 'mpl':
        # matplotlib style 
        # fig, ax = plt.subplots()
        # for i in dat.columns.levels[0]:
        #     df = dat[i]
        #     ax.plot(df['delta_time'],df[qty],label=i)
        # fig.autofmt_xdate()
        # ax.legend(loc='right')
        # ax.set_ylabel(qty)

        if not fig:
            fig, ax = plt.subplots(1,len(qty),figsize=(len(qty)*5,5))
            if len(qty) > 1:
                ax = ax.flat
            else: ax = [ax]
        else: ax = fig.axes
        dat.time = dat.time - dat.time.min()
        xlim = 0,dat.time.astype('timedelta64[s]').max()
        
        fig.autofmt_xdate()
        cmap = mpl.colormaps['tab20']
        cols = [ cmap(x) for x in np.linspace(0,1,20) ] 
        for j in dat['idx'].unique():
            df = dat[dat['idx'] == j]
            means = df.groupby(['t','idx',]).mean(numeric_only=False).reset_index(['t','idx'])
            std = df.groupby(['t','idx',]).std(numeric_only=True).reset_index(['t','idx'])
            maxm = df.groupby(['t','idx',]).max(numeric_only=True).reset_index(['t','idx'])
            #return df 
            for i,q in enumerate(qty):
                ax[i].set_title(f"{q} over time")
                extra_args = dict(label=j) if (i == len(qty) -1) else {}
                if stderr:
                    ax[i].errorbar(x=means['time'].astype('timedelta64[H]') ,y=means[q],yerr=std[q],color=cols[j%20],**extra_args)
                else: 
                    ax[i].plot(means['time'].astype('timedelta64[s]'),maxm[q],color=cols[j%20],**extra_args)
                    # Nicely format time
                    ax[i].xaxis.set_major_formatter(mticker.FuncFormatter(fmthours))

                    if xlim[1] > 3600 * 6: 
                        ax[i].xaxis.set_major_locator(mticker.MultipleLocator((xlim[1] // 3600 // 6)*3600) )
                    else: 
                        ax[i].xaxis.set_major_locator(mticker.MultipleLocator(120))

        ax[-1].legend(bbox_to_anchor=(1.00, 1.0),title= 'dot index')

        #fig.legend(loc=(0.87,0.25),title= 'dot index')
    elif style == 'plotly':
        from plotly.subplots import make_subplots

        cols = [ f'rgba{c}' for c in cols]
        # Multidimensional index b
        layout = dict(
                      autosize=True,
                      #width = 1000,
                      #height =1000,
                      xaxis_title="time (hours)",
                      #yaxis_title= q,
                      legend_title="Dots",
                     )
        n = len(qty)
        fig = make_subplots(rows=1, cols=n)#,layout=layout)
        for j in dat['idx'].unique():
            df = dat[dat['idx'] == j]
            means = df.groupby(['t','idx',]).mean().reset_index(['t','idx'])
            std = df.groupby(['t','idx',]).std().reset_index(['t','idx'])
            maxm = df.groupby(['t','idx',]).max().reset_index(['t','idx'])
            
            for i,q in enumerate(qty):
                extra_args = dict(showlegend=False) if i > 0 else {}
                if stderr:
                    error_y=dict(
                            type='data', # value of error bar given in data coordinates
                            array=std[q],
                            visible=True,),
                        
                else:
                    error_y = {}
                    
                fig.add_trace(go.Scatter(x=means['delta_time'] ,
                                         y=means[q] if stderr else maxm[q],
                                         error_y =error_y,
                                        # we group qties belonging to the same object 
                                        mode='lines', name=f"{j}",text=f"{j}", legendgroup=f"{j}",
                                        marker_color=cols[j%20], **extra_args
                                        ),
                             row=1,col=i+1, 
                             )
        fig.show()

# need to add keypoints 


# Format the seconds on the axis as min:sec
import matplotlib.ticker as mticker
def fmthours(x,pos):
    #return "{:02d}h{:02d}".format(int(x//3600), int((x%3600)//60)) 
    return "{:02d}h".format(int(x//3600), int((x%3600)//60)) 

def fmthours_nosymbol(x,pos):
    #return "{:02d}h{:02d}".format(int(x//3600), int((x%3600)//60)) 
    return "{:02d}".format(int(x//3600), int((x%3600)//60)) 

def time_to_td(time,axis=(1,2,3)):
        return (time - time[0]).mean(axis=axis).astype('timedelta64[s]') # time in seconds 

def plot_sh_regions(sh_stack,meta_stack,grid=None,dims_fig=[6,4],cbar=True,show_zbest=True,show_heatmap=True,show_absrange=True):
    import matplotlib.dates as mdates
    from datetime import datetime
    import pandas as pd 
    # Plot the location of the max sharpness slice, on top of the map of sharpness functions 
    n = sh_stack.shape[1]
    # fig,axs = plt.subplots(n,sharex=True,layout="tight")

    one_fig_dims = np.array(dims_fig) #np.array([10,6])

    if grid: ncol,nrow = grid
    else : ncol,nrow = 1,n
    
    fig, axs = plt.subplots( nrow,ncol,layout='constrained',sharex=True,figsize=one_fig_dims*[ncol,nrow],squeeze=False)
    # fig,axs = plt.subplots(n,sharex=True,layout="tight")
    # fig.set_figheight(n*2)
    zpos = get_xyz(meta_stack)[...,-1]
    time = get_time(meta_stack)
    timedeltas = time_to_td(time).astype(int)
    col_dz = 'darkgreen' #(1.0, 0., 0.6, 1.0) #bright purple color for the \delta_z plot

    # if len(axs)  == 1:
    #     axs = axs.flat
    # else:
    #     axs = np.array([axs])
    dat_best_sharpness =  []

    for i,ax in enumerate(axs.flat):
        if len(axs.flat) > 1 :
            ax.set_title(f'location \# {i}')
        ax.set_ylabel(r'$\mathrm{dz [\mu m]}$')#, ha='center', va='center', rotation='vertical')
        labs = [] # keep handles of plots somewhere to plot
        #ax.set_box_aspect(0.01)
        # 'auto' aspect is very important, otherwise gigantic white area appear 
        sh = sh_stack[:,i,0,:].T
        sh = (sh-sh.min())/(sh.max() - sh.min())
        #X,Y = np.meshgrid(timedeltas[:,i],np.arange(sh.shape[0]))
        xlim = timedeltas.min(),timedeltas.max()
        if show_heatmap:
            heatmap = ax.imshow(sh,aspect='auto',cmap='coolwarm',
                                #interpolation='none',
                                origin='lower',extent=(*xlim,0,sh.shape[0]))

        # Nicely format time
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmthours_nosymbol))
        # Use nice tick positions, let's say 6
        if xlim[1] > 3600 * 6: 
            ax.xaxis.set_major_locator(mticker.MultipleLocator((xlim[1] // 3600 // 6)*3600) )
        else: 
            ax.xaxis.set_major_locator(mticker.MultipleLocator(120))
        if cbar and show_heatmap: 
            fig.colorbar(heatmap,ax=ax,location='top',label='normalized sharpness')
        idx_z_best = np.argmax(sh,axis=0)
        nplanes = zpos.shape[-1]
        assert nplanes % 2 == 1 
        #z_midplane = zpos[:,i,0,:][:,nplanes//2]/
        #delta_z = (zpos[:,i,0,:] - z_midplane[:,None])[list(range(len(idx_z_best))),idx_z_best]
        dz = np.diff(zpos.flat[:2])[0]
        
        ticks = mticker.FuncFormatter(lambda x, pos: '{:2g}'.format((x-nplanes//2)*dz))
        #return idx_z_best,ticks
        #print(nplanes,dz)
        #return ticks
        ax.yaxis.set_major_formatter(ticks)
        #ax.yaxis.set_tick_params(labelcolor=col_dz)
        labs.append(ax.plot(timedeltas,idx_z_best,'-',color=col_dz,label='best plane $(\delta z)$'))
        #labs.append(ax.plot(timedeltas,delta_z,'-',color='purple',lw=3,label='best plane (dz)'))
        #return (-dz*(nplanes//2),dz*(nplanes//2))
        ax.set_ylim(0,nplanes-1)
        if show_zbest:
            ax2 = ax.twinx()
            ax2.set_ylabel(r'z [\unit{\micro\meter}]')
            #ax2.set_ylabel(r'$\delta z$ \[\unit{\micro\meter}\]$')#,color=col_dz)
            ax2.set_ylim(zpos[:,i,0,:].min(),zpos[:,i,0,:].max())
            z_best = zpos[:,i,0,:][list(range(len(idx_z_best))),idx_z_best]
            
            labs.append(ax2.plot(timedeltas,z_best,'-',color='black',label=' best z'))

            z_bottom = zpos[:,i,0,:][:,0]
            z_top = zpos[:,i,0,:][:,-1]
            #labs.append(ax2.plot(timedeltas,z_midplane,'-',color='black',label='z midplane'))
            if show_absrange:
                ax2.plot(timedeltas,z_bottom,'--',color='black')
                ax2.plot(timedeltas,z_top,'--',color='black')
            ax.legend(handles=[ x[0] for x in labs],loc='best')#,bbox_to_anchor=(1.20, 1.0))
        dat_best_sharpness.append((timedeltas,idx_z_best,i))

        #ax.autoscale(False)
        #ax2.autoscale(False)
    ax.set_xlabel('time [h]')
    
    # cax = fig.add_axes([ 0.05, 0.7, 0.0, 0.05, ])
    #if cbar: fig.colorbar(heatmap,ax=cax,location='bottom',label='normalized sharpness', pad=0.2)
    #fig.colorbar(heatmap, ax=axs,location='bottom',label='normalized sharpness')
    fig.autofmt_xdate()
    #plt.tight_layout(pzad=0, w_pad=0, h_pad=0)
    return fig,axs

def plot_sh_regions_simple(sh_stack,meta_stack,grid=None,dims_fig=[6,4]):
    import matplotlib.dates as mdates
    from datetime import datetime
    import pandas as pd 
    # Plot the location of the max sharpness slice, on top of the map of sharpness functions 
    n = sh_stack.shape[1]
    # fig,axs = plt.subplots(n,sharex=True,layout="tight")

    one_fig_dims = np.array(dims_fig) #np.array([10,6])
    
    fig, ax = plt.subplots(1,1,layout='constrained',sharex=True,figsize=one_fig_dims)

    zpos = get_xyz(meta_stack)[...,-1]
    time = get_time(meta_stack)
    timedeltas = time_to_td(time).astype(int)
    col_dz = 'darkgreen' #(1.0, 0., 0.6, 1.0) #bright purple color for the \delta_z plot

    dat_best_sharpness =  []
    npos = sh_stack.shape[1]
    #ax = axs
    # for i,ax in enumerate(axs.flat):
    #     if len(axs.flat) > 1 :
    #         ax.set_title(f'location \# {i}')
    ax.set_ylabel(r'$\mathrm{dz [\mu m]}$')#, ha='center', va='center', rotation='vertical')
    labs = [] # keep handles of plots somewhere to plot
    #ax.set_box_aspect(0.01)
    # 'auto' aspect is very important, otherwise gigantic white area appear 
    for i in range(npos):
        sh = sh_stack[:,i,0,:].T
        sh = (sh-sh.min())/(sh.max() - sh.min())
        #X,Y = np.meshgrid(timedeltas[:,i],np.arange(sh.shape[0]))
        xlim = timedeltas.min(),timedeltas.max()
        # Nicely format time
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmthours_nosymbol))
        # Use nice tick positions, let's say 6
        if xlim[1] > 3600 * 6: 
            ax.xaxis.set_major_locator(mticker.MultipleLocator((xlim[1] // 3600 // 6)*3600) )
        else: 
            ax.xaxis.set_major_locator(mticker.MultipleLocator(120))
        idx_z_best = np.argmax(sh,axis=0)
        nplanes = zpos.shape[-1]
        assert nplanes % 2 == 1 
        #z_midplane = zpos[:,i,0,:][:,nplanes//2]/
        #delta_z = (zpos[:,i,0,:] - z_midplane[:,None])[list(range(len(idx_z_best))),idx_z_best]
        dz = np.diff(zpos.flat[:2])[0]

        ticks = mticker.FuncFormatter(lambda x, pos: '{:2g}'.format((x-nplanes//2)*dz))
        #return idx_z_best,ticks
        #print(nplanes,dz)
        #return ticks
        ax.yaxis.set_major_formatter(ticks)
        #ax.yaxis.set_tick_params(labelcolor=col_dz)
        ax.plot(timedeltas,idx_z_best,'-',color=f'C{i}'.format(i),lw=3,label='\De$')
        #labs.append(ax.plot(timedeltas,delta_z,'-',color='purple',lw=3,label='best plane (dz)'))
        #return (-dz*(nplanes//2),dz*(nplanes//2))
        ax.set_ylim(0,nplanes-1)
        dat_best_sharpness.append((timedeltas,idx_z_best,i))
        print(i)
   #ax.autoscale(False)
   #ax2.autoscale(False)
    ax.set_xlabel('time [h]')
    
    # cax = fig.add_axes([ 0.05, 0.7, 0.0, 0.05, ])
    #if cbar: fig.colorbar(heatmap,ax=cax,location='bottom',label='normalized sharpness', pad=0.2)
    #fig.colorbar(heatmap, ax=axs,location='bottom',label='normalized sharpness')
    fig.autofmt_xdate()
    #plt.tight_layout(pzad=0, w_pad=0, h_pad=0)
    return fig,ax


def plot_zdot_and_zbeads(dat_dot,sh_stack,meta_stack,grid=None,cbar=True,figsize=(6,4)):
    import matplotlib.dates as mdates
    from datetime import datetime
    import pandas as pd 
    # Plot the location of the max sharpness slice, on top of the map of sharpness functions 
    n = sh_stack.shape[1]


    if grid: ncol,nrow = grid
    else : ncol,nrow = 1,n
    
    #fig, axs = plt.subplots( nrow,ncol,layout='constrained',sharex=True,figsize=one_fig_dims*[ncol,nrow],squeeze=False)
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True,layout='constrained',figsize=figsize)
    zpos = get_xyz(meta_stack)[...,-1]
    time = get_time(meta_stack)
    timedeltas = time_to_td(time).astype(int)
    col_dz = 'yellow' #(1.0, 0.0, 1.0, 1.0) #bright purple color for the \delta_z plot
    # for i,ax in enumerate(axs.flat):
    # # if len(axs.flat) > 1 :
    #     ax.set_title(f'location # {i}')
    # ax.set_ylabel('Z ($\mu$m)') 
    labs = [] # keep handles of plots somewhere to plot
    # 'auto' aspect is very important, otherwise gigantic white area appear 
    i = 0 
    sh = sh_stack[:,i,0,:].T
    sh = (sh-sh.min())/(sh.max() - sh.min())
    # get coordinates of sharpness heatmap
    X,_ = np.meshgrid(timedeltas,np.arange(sh.shape[0]))
    # the y value cannot be obtained by a meshgrid 
    Y = zpos[:,i].squeeze().T
    C = sh
    #heatmap = ax.pcolormesh(X,Y,C,cmap='coolwarm')
    if cbar: 
        fig.colorbar(heatmap,ax=ax,location='top',label='normalized sharpness', pad=0)

    xlim = timedeltas.min(),timedeltas.max()
    nplanes = zpos.shape[-1]
    assert (nplanes-1)%2 == 0,nplanes
    z_midplane = zpos[:,i,0,(nplanes-1)//2-1]
    # Nicely format time
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(fmthours_nosymbol))
    # Use nice tick positions, let's say every 6h
    if xlim[1] > 3600 * 6: 
        ax2.xaxis.set_major_locator(mticker.MultipleLocator((xlim[1] // 3600 // 6)*3600) )
    else: 
        ax2.xaxis.set_major_locator(mticker.MultipleLocator(120))

    # plot bead z 
    idx_z_best = np.argmax(sh,axis=0)
    z_best = zpos[:,i,0,:][list(range(len(idx_z_best))),idx_z_best]
    labs.append(ax.plot(timedeltas,z_best,'-',color='green',label='$z_{bead}$'))
    # plot offset plane 
    labs.append(ax.plot(timedeltas,z_midplane,'-',color='red',label='$z_{midplane}$'))
    # plot dot z
    time_dot = (dat_dot['time']-dat_dot['time'][0])/np.timedelta64(1, 's')
    z_dot = dat_dot['z']
    labs.append(ax2.plot(time_dot,z_dot,color='black',label='$z_{mark}$'))
    #ax.legend(handles=[ x[0] for x in labs])#,bbox_to_anchor=(1.20, 1.0))

    ax.autoscale(True)
    ax.set_xlim(xlim)
    ax2.set_xlabel(r'time [h]') # this cannot be moved

    # hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('none') 
    ax.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    #ax.set_ylabel('z ($\mu m$)',)
    #fig.text(0.05, 0.55, 'z [μm]', ha='center', va='center', rotation='vertical')
    # fig.set_constrained_layout_pads(w_pad=4./72., h_pad=4./72.,
    #         hspace=0./72., wspace=0./72.)
    ax.set_ylabel(r'$\mathrm{z [\mu m]}$', x=-10.0, y=-0.05)#, ha='center', va='center', rotation='vertical')
    #fig.tight_layout(rect=[0.0, 0.00, 1., 1.])
    #fig.autofmt_xdate()



def plot_dz_same_plot(sh_stack,meta_stack,tmax=None,grid=None,dims_fig=[6,4],cbar=True,show_zbest=True):
    '''Plot the timeseries of the  best focal plane offset for multiple dots
       returns: a list containing the timeseries (dz,time) for each of the dot (dims:n_dots*n_timepoints) 
    '''
    import matplotlib.dates as mdates
    from datetime import datetime
    import pandas as pd 
    # Plot the location of the max sharpness slice, on top of the map of sharpness functions 
    n = sh_stack.shape[1]
    # fig,axs = plt.subplots(n,sharex=True,layout="tight")
    cmap = mpl.colormaps['tab20']
    cols = [ cmap(x) for x in np.linspace(0,1,20) ] 
    marks =  ('.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
    one_fig_dims = np.array(dims_fig) #np.array([10,6])


    if grid: ncol,nrow = grid
    else : ncol,nrow = 1,n 
    
    fig, ax = plt.subplots(layout='constrained',sharex=True,figsize=one_fig_dims,squeeze=True)
    # fig,axs = plt.subplots(n,sharex=True,layout="tight")
    # fig.set_figheight(n*2)
    zpos = get_xyz(meta_stack)[...,-1]
    time = get_time(meta_stack)
    timedeltas = time_to_td(time).astype(int)
    #col_dz = 'yellow' #(1.0, 0.0, 1.0, 1.0) #bright purple color for the \delta_z plot
    # if len(axs)  == 1:
    #     axs = axs.flat
    # else:
    #     axs = np.array([axs])
    dat_best_sharpness =  []

    for i,col,mark in zip(range(n),cols,marks):
        ax.set_ylabel(r'$\mathrm{dz [\mu m]}$')#.set_ylabel('δz [μm]')
        labs = [] # keep handles of plots somewhere to plot
        #ax.set_box_aspect(0.01)
        # 'auto' aspect is very important, otherwise gigantic white area appear 
        sh = sh_stack[:,i,0,:].T
        sh = (sh-sh.min())/(sh.max() - sh.min())
        #X,Y = np.meshgrid(timedeltas[:,i],np.arange(sh.shape[0]))
        
        xlim = [timedeltas.min(),timedeltas.max()]
        if tmax:
            xlim[1] = tmax
        #heatmap = ax.imshow(sh,aspect='auto',cmap='coolwarm',interpolation='none',origin='lower',extent=(*xlim,0,sh.shape[0]))

        # Nicely format time
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmthours_nosymbol))
        # Use nice tick positions, let's say 6
        if xlim[1] > 3600 * 6: 
            ax.xaxis.set_major_locator(mticker.MultipleLocator((xlim[1] // 3600 // 6)*3600) )
        else: 
            ax.xaxis.set_major_locator(mticker.MultipleLocator(120))
        #if cbar: fig.colorbar(heatmap,ax=ax,location='left',label='normalized sharpness', pad=0.2)
        idx_z_best = np.argmax(sh,axis=0)
        nplanes = zpos.shape[-1]
        assert nplanes % 2 == 1 
        #z_midplane = zpos[:,i,0,:][:,nplanes//2]
        #delta_z = (zpos[:,i,0,:] - z_midplane[:,None])[list(range(len(idx_z_best))),idx_z_best]
        dz = np.diff(zpos.flat[:2])[0] # assumes that dz is constant 
        ticks = mticker.FuncFormatter(lambda x, pos: '{:2g}'.format((x-nplanes//2)*dz))
        ax.yaxis.set_major_formatter(ticks)
        ax.plot(timedeltas,idx_z_best,color=f'C{i}',alpha=0.5,label=r'$\Delta z ({})$'.format(i))
        #labs.append(ax.plot(timedeltas,delta_z,'-',color='purple',lw=3,label='best plane (dz)'))
        #return (-dz*(nplanes//2),dz*(nplanes//2))
        ax.set_ylim(0,nplanes-1)
        ax.autoscale(True)
        ax.set_xlim(xlim)
        dat_best_sharpness.append(dz*(idx_z_best-(nplanes-1)/2))
        #dat_best_sharpness.append(idx_z_best)
    ax.legend(loc=(1.1,0.05),handlelength=0.1)#handles=[ x[0] for x in labs])#
    ax.set_xlabel(r'time [h]')
    ax.set_ylim(0,nplanes-1)
    #fig.autofmt_xdate()
    return fig,ax, dat_best_sharpness


def compute_sh_stack(folder,func=None):
    ''' Takes a folder, and applies a function on all images, then returns a ndarray with this function applied'''
    from tqdm.auto import tqdm
    from mmpyreader import MMpyreader
    import json
    reader =  MMpyreader()
    stack = reader.load_folder(folder)

    dims = tuple(stack.dims.values())
    keys = stack.dims.keys()
    sh_stack = np.zeros(dims)
    meta_stack = np.zeros(dims,dtype=object)
    
    if not func:
        func = redondo

    for i in tqdm(np.ndindex(dims),total=np.prod(dims)):
        try:
            img,meta = stack.get_img(**dict(zip(keys,i)))
            sh_stack[i] = func(img).sum()
            meta_stack[i] = json.loads(meta.toPropertyMap().toJSON())
        except: 
            pass
        #cnt += 1 
        #if cnt == 21: break
        
    return sh_stack,stack,meta_stack




# %%
def plot_locations(meta_stack):
    xs,ys,zs = get_xyz(meta_stack)[0,:,0,0].T
    labs = np.vectorize(lambda X: np.array(X['map']['PositionName']['scalar']), signature='()->()')(meta_stack)
    labs = labs[0,:,0,0]
    plt.scatter(xs,ys)
    for lab,x,y in zip(labs,xs,ys):
        plt.annotate(lab,(x+50,y-50),)
# %%

def plot_pos(inp,labels=True,color=None,fig=None,cbar=True):
    ''' Plot position of dots, either from a .pos file, or from a metadata stack file'''
    if type(inp) == np.ndarray: 
        xs,ys,zs = get_xyz(inp)[0,:,0,0].T
        labs = np.vectorize(lambda X: np.array(X['map']['PositionName']['scalar']), signature='()->()')(inp)
        labs = labs[0,:,0,0]
    else:
        import json
        assert inp.endswith('.pos')
        with open(inp,'r') as f:
            res = json.load(f)

        pos = [pos['DevicePositions']['array'][-1]['Position_um']['array'] for pos in res['map']['StagePositions']['array'] ] 
        labs = [pos['Label']['scalar'] for pos in res['map']['StagePositions']['array'] ] 
        pos_dict = dict(zip(labs,pos))
    
        xs,ys = np.array(pos).T
        xs = -xs

    if not color:
        color = ['red' if 'dot' in lab.lower() else 'blue' for lab in labs]
    else:
        color = color[:len(xs)]
    if not fig:    
        fig,ax = plt.subplots(1,layout='constrained')
    else:
        ax = fig.axes[0]
    sc = ax.scatter(xs,ys,color=color)
    
    # add colorbar
    if cbar:
        bounds = np.arange(0,len(color)+1) #+ [len(colors)]
        cmap = mpl.colors.ListedColormap(color)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)#, extend='both')
    
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                     orientation='vertical',ax=ax
                     #label="Discrete intervals with extend='both' keyword"
                     )
        cbar.set_ticks(bounds[:-1]+0.5,labels=bounds[:-1])
        cbar.minorticks_off()
        cbar.ax.tick_params(right=False)
 
    #cbar = fig.colorbar(color, ax=ax)
    if labels:
        for lab,x,y in zip(labs,xs,ys):
            color = 'red' if 'dot' in lab.lower() else None
            plt.annotate(lab,(x+50,y-50),color=color)
    plt.xlabel(r'$\mathrm{x [\mu m]}$')
    plt.ylabel(r'$\mathrm{y [\mu m]}$')
    return fig,ax


def plot_sh_as_lines(sh,one_fig_dims=np.array([8*cm,6*cm])):
    # plot the sharpness functions over time 
    from matplotlib import colormaps
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from math import ceil
    cmap = colormaps['viridis']
    ts,ps = sh.shape[:2]
    ncol = 2
    nrow = ceil(ps/ncol)
    fig, ax = plt.subplots( nrow,ncol,layout='tight',sharex=True,figsize=one_fig_dims*[ncol,nrow])
    #fig.set_figwidth(20)
    ax = ax.flat
    for p in range(ps):
        for t in range(ts): 
            dat = (sh[t,p,0,:] - sh[t,p,0,:].min())/(sh[t,p,0,:].max() - sh[t,p,0,:].min())
            ax[p].plot(dat,color=cmap(float(t/ts))) 
            #ax[p].set_box_aspect(1)
            ax[p].set_title(f'dot {p}',)
        #cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap,norm=Normalize(vmin=0,vmax=ts)), ax=ax[p])
        #cbar.set_label('time')

    #fig.tight_layout()
    return fig 



# These functions should not be use
def compute_sh_stack_parallel(folder):
    ''' Q: can ZMQ handle parallel tasks ? NO
        this will fail at some point 
        >>> out = compute_sh_stack_parallel('/mnt/dioscuri-nas/Ilyas/Microscope/23-04-27_imperfect_focus_bench_with_beads_2/x20_PFS_dz0,5_1/')
    '''

    raise Exception("This function doesn't work")
    from tqdm.auto import tqdm
    from mmpyreader import MMpyreader
    import dask
    reader =  MMpyreader()
    stack = reader.load_folder(folder)

    dims = tuple(stack.dims.values())
    keys = stack.dims.keys()
    sh_stack = np.zeros(dims)
    meta_stack = np.zeros(dims,dtype=object)
    out = []
    for i in list(np.ndindex(dims))[:100]:
        ret = dask.delayed(stack.get_img)(**dict(zip(keys,i)))
        ret = dask.delayed(redondo)(ret[0])
        ret = dask.delayed(np.sum)(ret)
        out.append(ret)
    return dask.compute(*out)

def compute_acq_time(meta):
    times =  get_time(meta).flat
    return times[-1] - times[0]


def plot_all_sharpnesses(sh_all,dx=None):
    # I "normalize" 
    if dx:
        plt.xlabel(r'$\mu$m')
    plt.ylabel(r'$\bar{S}(z)$')
    out = [] 
    for i,sh in enumerate(sh_all):
        c = np.argmax(sh)
        x = np.arange(len(sh))
        x = x - x[c]
        y = (sh-min(sh))/(max(sh)-min(sh))
        if dx:
            x = x*dx
        plt.plot(x,y)
        out.append([x,y])
    return out
    
def plot_sharpness_profile(path,tref=-1):
    ''' plots the sharpness profile fro all dots at a given timepoint
        args: 
            -path location of the folder containg the sharpness los
            - tref reference time (counted from the end) 
    '''
    dat_sh = parse_data_sh(path)
    fig, ax = plt.subplots()
    
    tref = dat_sh[0]['t'].max() + tref
    # go through all dots
    for e in dat_sh.columns.levels[0]:
        # select events at that timepoint
        idxs = dat_sh[e]['t'] == tref
        # retrieve sharpness
        sh = dat_sh[e]['sh'][idxs]
        #retrieve z
        z = dat_sh[e]['z'][idxs]
        # center z
        z -= z.mean()
        sh_normalised = (sh-sh.min())/(sh.max()-sh.min()) 
        ax.plot(z,sh_normalised,label=e)  
    ax.set_xlabel('z');
    ax.set_ylabel(r'$\bar{S} \left[a.u.\right]$');
    return fig

            



def plot_images_in_grid(img,scale=3,ncol=2):
    n = len(img)
    shape = np.array([n//ncol,ncol]) #y, then x 
    fig, ax = plt.subplots(*shape,figsize=scale*shape[::-1]*img[0].shape[::-1]/img[0].shape[0],layout='tight')
    ax = ax.ravel()
    
    for i in range(len(ax)):
        ax[i].imshow(img[i],cmap = 'gray')
        ax[i].set_xticks([]), ax[i].set_yticks([])               
        ax[i].text(50,300,str(i),fontsize=48,c=glasbey[i])   

# def plot_images_in_grid(img,scale=3,ncol=2):
#     n = len(img)
#     shape = np.array([n//ncol,ncol]) #y, then x 
#     fig, ax = plt.subplots(*shape,figsize=scale*shape[::-1]*img[0].shape[::-1]/img[0].shape[0],layout='tight')
#     ax = ax.ravel()
    
#     for i in range(len(ax)):
#         ax[i].imshow(img[i],cmap = 'gray')
#         ax[i].set_xticks([]), ax[i].set_yticks([])             
        


def one_bin_per_value(zvals):
    '''small function that takes a set of value and compute bins edges such that each value has its own bin'''
    step=np.unique(np.diff(zvals)) # this is not really necessary
    bins = (zvals[:-1] + zvals[1:])/2
    bins = np.insert(bins,0,bins[0]-step[0])
    bins = np.append(bins,bins[-1]+step[-1]) 
    return bins 



def get_sh_all_dots(folder,channel=None):
    '''Compute the sharpness on a multi-dimensional stack like how it's done
       during a microscopy acquisition (e.g. cropped in the center) 
    ''' 
    import dask
    import json
    from mltools.utils_images import sharpness
    from mmpyreader import MMpyreader
    with MMpyreader() as reader:
        stack = reader.load_folder(folder)
        stack = stack.get_substack(C=channel)

    
        imgs,meta = stack[0].squeeze(),stack[1].squeeze()
        meta = np.vectorize(lambda x: json.loads(x.toPropertyMap().toJSON()))(meta)
    if imgs.ndim == 3: 
        imgs = imgs[None,...]
        meta = meta[None,...]
        
    sh_all = np.zeros(imgs.shape[:2])
    imgs_best = []
    meta_best = []
    for i,(pos,sh_current) in enumerate(zip(imgs,sh_all)):
        h,w = (1200,1920)
#        window = [h/2-0.1*h:h/2+0.1*h,w/2-0.1*w:w/2+0.1*w]       
        sh = [dask.delayed(sharpness)(img[h//2-h//10:h//2+h//10,w//2-w//10:w//2+w//10]) for img in pos]
        sh = dask.compute(*sh)    
        sh_current[:] = sh[:]
        zbest = np.argmax(sh)
        imgs_best.append(pos[zbest])
        #meta_best.append(meta[i][zbest])
        
    imgs_best = np.array(imgs_best)
#    meta_best = np.array(meta_best,dtype=object)
    
    return imgs_best,meta,sh_all 


def plot_dz_hist(all_dz,ax=None):

    # Bin every positions 
    if not ax:
        fig,ax = plt.subplots(figsize=(8*cm,16/3*cm),layout='constrained')
    all_dz = np.array(all_dz).flatten()
    # make bins centered around midpoints between actual values
    zvals,cnts = np.unique(all_dz,return_counts=True)
    
    
    bins = one_bin_per_value(zvals) 
    cnts, _ = np.histogram(all_dz,bins)
    cnts = cnts/len(all_dz) #get frequencies instead of "density"
    ax.stairs(cnts,bins,label='',color='green',fill=True)
    #plt.hist(all_dz,density=True)
    ax.set_xlabel(r'$\mathrm{dz}$  [$\mu$m]')
    ax.set_ylabel('frequency');
    
    return all_dz

def plot_and_compute_fwhm(all_dz,ax,bandwidth=0.5,linewidth=3.0):
    from sklearn.neighbors import KernelDensity
    from scipy.signal import find_peaks, peak_widths

    # make bins centered around midpoints between actual values
    zvals,cnts = np.unique(all_dz,return_counts=True)
    
    
    bins = one_bin_per_value(zvals) 
    cnts, _ = np.histogram(all_dz,bins)
    cnts = cnts/len(all_dz) #get frequencies instead of "density"
    ax.stairs(cnts,bins,label='',color='green',fill=True)
    #plt.hist(all_dz,density=True)
    ax.set_xlabel(r'$\mathrm{dz} \left[\mathrm{\mu m} \right]$')
    ax.set_ylabel('frequency');
    norm_factor = sum(np.diff(bins)*cnts)
    
    # mere attempt to fit the observations to a distribution
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(all_dz[:, np.newaxis])
    bnds = np.min(all_dz),np.max(all_dz)
    x = np.linspace(*bnds,100)[:, np.newaxis]
    log_dens = kde.score_samples(x)
    X,Y = x[:,0],np.exp(log_dens)*norm_factor
    ax.plot(X,Y,label='distribution estimate',alpha=1)


    # Now compute the FWHM
    peaks, _ = find_peaks(Y)
    peak_widths(X,peaks, rel_height=0.5, prominence_data=None, wlen=None)
    main_peak = [peaks[np.argmax(Y[peaks])]]
    results_half = peak_widths(Y, main_peak, rel_height=0.5)
    half_height,x0,xend = results_half[1:]
    x0,xend = np.interp(results_half[2:],np.arange(0,len(X)),X)
    fwhm = (xend-x0)
    
    ax.hlines(half_height,x0,xend, color="black",linewidth=linewidth)
    # plt.hlines(*results_full[1:], color="C3"))
    #plt.show()
    
    print(x0,xend,fwhm)
    #from sklearn.neighbors import KernelDensity
    #norm_factor = sum(np.diff(bins)*cnts)
    
    # mere attempt to fit the observations to a distribution
    # kde = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(all_dz[:, np.newaxis])
    # x = np.linspace(-3,3,100)[:, np.newaxis]
    # log_dens = kde.score_samples(x)
    # #plt.fill(x[:, 0], np.exp(log_dens), fc="#AAAAFF")
    # plt.plot(x[:,0],np.exp(log_dens)*norm_factor,label='distribution estimate',alpha=1)
    # # normalization factor:

    
