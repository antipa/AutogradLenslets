import autograd.numpy as np
import autograd.scipy as sc
import matplotlib.pyplot as plt
import copy
from IPython import display
from os import listdir
from os.path import isfile, join
from scipy import signal
from skimage.transform import resize as imresize
from skimage.restoration import (denoise_wavelet, estimate_sigma)
import scipy.misc as misc
import autograd.scipy.signal

def imnormalize(x):
    return x/np.max(x)

def read_im_list_rgb(folder, files, samples, window = False):
    return_list = []
    if window:
        im_window = np.atleast_3d(np.outer(signal.windows.tukey(samples[0]*2, .7), signal.windows.tukey(samples[1]*2, .7)))
    else:
        im_window = np.atleast_3d(np.ones(samples))
    [return_list.append(np.moveaxis(im_window*imresize(imnormalize(misc.imread(folder+files[n]).astype('float32')), tuple(2*samples[n] for n in range(len(samples)))), -1, 0))  for n in range(len(files))]
    return return_list

def make_lenslet_surface_ag(Xlist, Ylist, Rlist, xg, yg, offset=10., mode='radius'):
    # Takes in Xlist, Ylist and Rlist: floating point center and radius values for each lenslet
    # xrng and yrng: x and y range (tuple) over which to define grid
    # samples: tuple of number of samplex in x and y, respectively
    #
    # Outputs: T, the aperture thickness function. 
    
    # If only one radius provided, use same radius for everything
    if np.shape(Rlist) == ():
        Rlist = np.ones_like(Xlist)*Rlist
    if mode is 'curvature':
        Rlist = [1/Rlist[n] if Rlist[n] != 0 else 1/(Rlist[n]+1e-5) for n in range(len(Rlist))]
        
    Nlenslets = Xlist.shape[0]
    T = np.zeros_like(xg)
    for n in range(Nlenslets):

        sph = np.real(np.sqrt(0j+Rlist[n]**2 - (xg-Xlist[n])**2 - (yg-Ylist[n])**2))-Rlist[n]+offset
        T = np.maximum(T,sph)
    
    return T-offset


def gen_psf_ag(surface, ior, t, z_obj, obj_def, field, CA, lmbda, xg, yg, Fx, Fy,pupil_phase=0, prop_pad = 0):
    # Inputs:
    # surface: single surface thickness function, units: mm
    # ior : index of refraction of bulk material
    # t : thickness of surface (i.e. distance to output plane)
    # z_obj : distance from object plane. +Inf means object at infinity
    # object_def : 'angle' for angular field definition, 'obj_height' for finite
    # field : tuple (x,y) wavefront field definition. (0,0) is on-axis. Interpreted in context of object_def
    # CA: radius of clear aperture in mm
    # pupil_aberration: additional pupil phase, in radians!
    # lmbda: wavelength in mm
    # xg and yg are the spatial grid (pixel spacing in mm)
    # Fx : frequency grid in 1/mm
    # Fy : same as Fx
    k = np.pi*2/lmbda
    
    if obj_def is 'angle':
        ramp_coeff_x = -np.tan(field[0]*np.pi/180)
        ramp_coeff_y = -np.tan(field[1]*np.pi/180)
        ramp = xg*ramp_coeff_x + yg*ramp_coeff_y
        if z_obj is 'inf':
            U_in = np.exp(1j*k*(ramp))
        else:
            U_in = np.exp(1j*k*(-z_obj*np.sqrt(1-(xg/z_obj)**2 - (yg/z_obj)**2) + ramp))
    elif obj_def is 'obj_height':
        if z_obj is 'inf':
            raise Exception('cannot use obj_height and object at infinity')
        else:
            U_in = np.exp(1j*-z_obj*k*np.sqrt(1-((xg-field[0])/z_obj)**2 - ((yg-field[1])/z_obj)**2))
    
    U_out = U_in * np.exp(1j*(k*(ior-1)*surface + pupil_phase))
    amp = np.sqrt(xg**2 + yg**2) <= CA
    U_prop = propagate_field_freq(lmbda, t, amp*U_out, Fx, Fy)
    
    psf = np.abs(U_prop)**2
    return(psf/np.sum(psf))
    

def propagate_field_freq(lam, z, U, Fx, Fy, padfrac=0):
    k = 2*np.pi/lam

    #siz = np.shape(U)
    #fx = np.linspace(-1/2/ps,1/2/ps,siz[1])
    #fy = np.linspace(-1/2/ps,1/2/ps,siz[0])
    #x = np.linspace(-siz[1]/2*ps,siz[1]/2*ps,siz[1])
    #y = np.linspace(-siz[0]/2*ps,siz[0]/2*ps,siz[0])
    #X,Y = np.meshgrid(x,y)
    #Fx,Fy = np.meshgrid(fx,fy)
    if padfrac != 0:
        shape_orig = np.shape(U)
        U = pad_func(U, padfrac, 'edge')

        Fx, Fy = np.meshgrid(np.linspace(np.min(Fx), np.max(Fx), U.shape[0]), np.linspace(np.min(Fy), np.max(Fy), U.shape[1]))
        
    Uf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U)))
    Hf = np.exp(1j*2*np.pi*z/lam * np.sqrt(1-(lam*Fx)**2 - (lam*Fy)**2))
    Up = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Uf*Hf)))
    if padfrac != 0:
        Up = crop_func(Up, shape_orig)
    return Up    



def forward_sim(psf, im, padfrac=1/2, crop_size = (0, 0), noise_var = 1):
    
    #if psf.dtype is np.dtype('float64'):
    if crop_size == (0,0):
        if psf.ndim < im.ndim:
            psf = np.expand_dims(psf, 0)
            if crop_size == (0,0):
                crop_size = (3, psf.shape[1], psf.shape[2])

        else:
            crop_size = psf.shape
#    psf_shape = np.shape(psf)

    
    H = np.fft.fft2(np.fft.ifftshift(pad_func(psf, padfrac)))
    
    if np.shape(im)[-2] != np.shape(H)[-2]:
        im = pad_func(im, padfrac)
        #noise_var*np.random.randn(np.shape(psf)[-2],np.shape(psf)[-1])+
#     print("Padded psf shape is:")
#     print(np.shape(H))
    
#     print("Im shape is:")
#     print(np.shape(im))
    filtered = np.real(np.fft.ifft2(H*np.fft.fft2(im)))
#     print("Convolved shape:")
#     print(np.shape(filtered))
    cropped = crop_func(filtered, crop_size)
#     print("Cropped shape is:")
#     print(np.shape(cropped))
    return(cropped)

def project_to_aperture(x_list, y_list, aperR, mode='snap'):
    
    lr = np.sqrt(x_list**2+y_list**2)
    lout = lr<aperR
    if mode is 'delete':
        x_out = x_list[lout]
        y_out = y_list[lout]
    elif mode is 'snap':
        ya = np.arctan2(y_list, x_list)
        lr = np.minimum(lr, aperR)
        y_out = lr * np.sin(ya)
        x_out = lr * np.cos(ya)
        
    return x_out, y_out

def rgb2imshow(im):
    return imnormalize(np.maximum(np.moveaxis(im,0,-1),0))

def get_files(folder):
    return([f for f in listdir(folder) if isfile(join(folder, f))])

def load_training_ims(samples, dataset_dir, Ntrain, Nbatch, Nval):
    imfiles = get_files(dataset_dir)  #Get list of files
    Nfiles = len(imfiles)
    train_inds = np.random.choice(Nfiles, Ntrain, replace=False)    #Generate random indices for training data
    train_files = [imfiles[n] for n in train_inds]     #Get list of training files associated with indices
    [imfiles.pop(n) for n in sorted(train_inds, reverse=True)]    #remove these files from the list


    val_inds = np.random.choice(Nfiles-Ntrain, Nval, replace=False)   #Generate random indices for validation from the remaining files
    val_files = [imfiles[n] for n in val_inds]            # Get file list from indices
    
    train_list = read_im_list_rgb(dataset_dir, train_files, samples, window=True)   #Load training data
    val_list = read_im_list_rgb(dataset_dir, val_files, samples, window=True)    #Load validation data
    return train_list, val_list



# Operate on last 2 dims to match np.fft convention
def pad_func(x, padfrac):
    if np.shape(padfrac) == ():
        if x.ndim is 2:
            padfrac = ((padfrac, padfrac), (padfrac, padfrac))
        elif x.ndim is 3:
            #If x is 3D and pad a single pad value was passed in, assume padding on last 2 dims only
            padfrac = ((0,0), (padfrac, padfrac), (padfrac, padfrac))
        
        
    padr = [];
    for n in range(x.ndim):
        pwpre = np.ceil(padfrac[n][0]*x.shape[n]).astype('int')
        pwpost = np.ceil(padfrac[n][1]*x.shape[n]).astype('int')
        padr.append((pwpre,pwpost))
        #x = zero_pad_ag(x, padr[n], n-2)
    #print("Padr:")
    #print(padr)
    return np.pad(x,padr,'constant')

def crop_func(x,crop_size):
    # Crops the center matching the size in the tuple crop_size. Implicitly deals with higher dimensions?
    cstart = []
    cent = []
    for n in range(x.ndim):
        cstart.append((x.shape[n]-crop_size[n])//2)
    slicer = tuple(slice(cstart[n],cstart[n]+crop_size[n],1) for n in range(len(crop_size)))
    return(x[slicer])


def zero_pad_ag(x, pad_tuple, pad_dim):
    zarg_pre = tuple(pad_tuple[0] if z == pad_dim % x.ndim else x.shape[z] for z in range(x.ndim))
    zpre = np.zeros(zarg_pre)
    zarg_post = tuple(pad_tuple[1] if z == pad_dim % x.ndim else x.shape[z] for z in range(x.ndim))
    zpost = np.zeros(zarg_post)
    out = np.concatenate((zpre, x, zpost), axis=pad_dim)
    
    return out

def fftconv(x, H):
    return np.real(np.fft.ifft2(H*np.fft.fft2(x)))

def admm2d(y, psf_in, tau, niter, options = 'default'):
    #Inputs:
    # y : measurement
    # psf_in : point spread function
    # tau : regularizer
    # niter : number of ADMM iterations
    # options : options dict generated by gen_options. If left off, default will be created.
    #def admm_unrolled(b, psf, tau, iters):
    if options is 'default':
        options = gen_options()

    mu1 = options["mu1"]
    mu2 = options["mu2"]
    mu3 = options["mu3"]
    disp_interval = options["disp_interval"]
    resid_tol = options["resid_tol"]
    mu_inc = options["mu_inc"]
    mu_dec = options["mu_dec"]
    pad_frac = options["pad_frac"]
    if disp_interval != 0:
        fig1, ax = plt.subplots(2,3, figsize=options["fig_size"])
    Ny, Nx = psf_in.shape
    pad = lambda x:pad_func(x,pad_frac)
    crop = lambda x:crop_func(x,np.shape(psf_in))
    H = np.fft.fft2(np.fft.ifftshift(pad(psf_in)))
    H_conj = np.conj(H)
    #Hfor = lambda x:np.real(np.fft.ifft2(H*np.fft.fft2(x)))
    Hfor = lambda x:fftconv(x,H)
    Hadj = lambda x:fftconv(x, H_conj)  #np.real(np.fft.ifft2(H_conj*np.fft.fft2(x)))

    Cty = pad(y)
    sk = np.zeros_like(Cty)
    alpha1k = np.zeros_like(Cty)
    alpha3k = np.zeros_like(Cty)
    alpha3kp = np.zeros_like(Cty)
    L = lambda D:(-np.diff(D,axis=0),-np.diff(D,axis=1))
    Ltv = lambda P1,P2:np.vstack([P1[0,:],np.diff(P1,axis=0),-P1[-1,:]]) + np.hstack((P2[:,[0]],np.diff(P2,axis=1),-P2[:,[-1]]))
    lapl = np.zeros_like(sk)
    lapl[0,0]=4
    lapl[0,1]=-1
    lapl[1,0]=-1
    lapl[0,-1]=-1
    lapl[-1,0]=-1
    LtL = np.abs(np.fft.fft2(lapl))
    alpha2k_1 = copy.deepcopy(sk[:-1,:])
    alpha2k_2 = copy.deepcopy(sk[:,:-1])
    HtH = np.abs(H*H_conj)
    Smult = 1/(mu1*HtH + mu2*LtL + mu3)
    CtC = pad(np.ones_like(y))
    Vmult = 1/(CtC + mu1)
    Hskp = np.zeros_like(Vmult)
    dual_resid_s = []
    primal_resid_s = []
    dual_resid_u = []
    primal_resid_u = []
    dual_resid_w = []
    primal_resid_w = []
    cost = []
    ukp_1, ukp_2 = L(np.zeros_like(y))
    Lsk1 = ukp_1
    Lsk2 = ukp_2

    for n in range(niter):

        Lsk1, Lsk2 = L(sk)

        ukp_1, ukp_2 = soft_2d_gradient(Lsk1 + alpha2k_1/mu2, Lsk2 + alpha2k_2/mu2, tau/mu2)
        #Hsk = copy.deepcopy(Hskp);   ##### DEEPCOPY!?
        Hsk = Hskp
        vkp = Vmult*(mu1*(alpha1k/mu1 + Hsk) + Cty)
        wkp = np.maximum(alpha3k/mu3 + sk, 0)
        skp_numerator = mu3*(wkp - alpha3k/mu3) + mu2*Ltv(ukp_1 - alpha2k_1/mu2, ukp_2 - alpha2k_2/mu2) + mu1 * Hadj(vkp - alpha1k/mu1)
        skp = np.real(np.fft.ifft2(Smult * np.fft.fft2(skp_numerator)))

        Hskp = Hfor(skp)
        r_sv = Hskp - vkp
        dual_resid_s.append(mu1 * np.linalg.norm(Hsk - Hskp,'fro'))
        primal_resid_s.append(np.linalg.norm(r_sv,'fro'))

        mu1, mu1_update = update_param(mu1, resid_tol, mu_inc, mu_dec, primal_resid_s[-1], dual_resid_s[-1])
        alpha1k += mu1*r_sv

        Lskp1, Lskp2 = L(skp)
        r_su_1 = Lskp1 - ukp_1
        r_su_2 = Lskp2 - ukp_2
        dual_resid_u.append(mu2*np.sqrt(np.linalg.norm(Lsk1 - Lskp1,'fro')**2 + np.linalg.norm(Lsk2 - Lskp2,'fro')**2))
        primal_resid_u.append(np.sqrt(np.linalg.norm(r_su_1,'fro')**2 + np.linalg.norm(r_su_2,'fro')**2))


        mu2, mu2_update = update_param(mu2, resid_tol, mu_inc, mu_dec, primal_resid_u[-1], dual_resid_u[-1])
        alpha2k_1+= mu2*r_su_1
        alpha2k_2+= mu2*r_su_2


        r_sw = skp - wkp
        dual_resid_w.append(mu3*np.linalg.norm(sk - skp,'fro'))
        primal_resid_w.append(np.linalg.norm(r_sw,'fro'))

        mu3, mu3_update = update_param(mu3, resid_tol, mu_inc, mu_dec, primal_resid_w[-1], dual_resid_w[-1])
        alpha3k += mu3*r_sw

        mu_update = 0
        if mu1_update or mu2_update or mu3_update:
            mu_update = 1
            Smult = 1/(mu1*HtH + mu2*LtL + mu3)
            Vmult = 1/(CtC + mu1)

        sk = skp;
        cost.append(np.linalg.norm(crop(Hskp)-y,'fro')**2 + tau*TVnorm(skp))
        if disp_interval != 0:
            if n % disp_interval == 0: 


                ax[0,0].imshow(sk)
                ax[0,0].set_title('cost. Mu update: {}'.format(mu_update))


                ax[0,1].semilogy(cost)
                ax[0,1].set_title('cost')
                
                ax[0,2].semilogy(dual_resid_s)
                ax[0,2].semilogy(primal_resid_s)
                ax[0,2].set_title('crop (s) residuals, mu1 = %.2f' %mu1)

                ax[1,0].semilogy(dual_resid_u)
                ax[1,0].semilogy(primal_resid_u)
                ax[1,0].set_title('TV (u) residuals, mu2 = %.2f'  % mu2)

                ax[1,1].semilogy(dual_resid_w)
                ax[1,1].semilogy(primal_resid_w)
                ax[1,1].set_title('Nonnegativity (w) residuals, mu3 = %.2f'  % mu3)

                ax[1,2].imshow(psf_in)
                ax[1,2].set_title('psf')

                display.display(fig1)
                display.clear_output(wait=True)
    return sk
        
        

    
def soft_2d_gradient(v,h,tau):

    mag = np.sqrt(np.vstack([v,np.zeros((1,v.shape[1]))])**2 + np.hstack((h,np.zeros((h.shape[0],1))))**2)
    magt = np.maximum(mag - tau,0)
    mag = np.maximum(mag - tau, 0) + tau
    mmult = magt/mag
    #mmult[mag==0] = 0
    return v*mmult[:-1,:], h*mmult[:,:-1]

def update_param(mu, resid_tol, mu_inc, mu_dec, r, s):
    if r > resid_tol * s:
        mu_out = mu*mu_inc
        mu_update = 1
    elif r*resid_tol < s:
        mu_out = mu/mu_dec
        mu_update = -1
    else:
        mu_out = mu
        mu_update = 0
    return mu_out, mu_update

def TVnorm(x):
    result = 0
    for n in range(x.ndim):
        result += np.sum(np.abs(np.diff(x,axis=n)))
    return result

def gen_options(mu1 = .8, mu2 = .22, mu3 = .25, disp_interval = 0, resid_tol = 1.5, mu_inc = 1.2, mu_dec = 1.2, pad_frac = 1/2, fig_size = plt.rcParams.get('figure.figsize')):
    out = {
        "mu1" : mu1,
        "mu2" : mu2,
        "mu3" : mu3,
        "disp_interval" : disp_interval,
        "resid_tol" : resid_tol,
        "mu_inc" : mu_inc,
        "mu_dec" : mu_dec,
        "pad_frac" : pad_frac,
        "fig_size" : fig_size
    }
    return out




def FISTA(grad_func, proj_func, x_init, mu, niter, do_restarting = True):
    y_k = x_init
    x_k = x_init
    t_k = 1
    f_out = []
    for n in range(niter):    
        g, f = grad_func(y_k)
        
        x_kp = proj_func(y_k - mu * g, n)
        t_kp = (1 + np.sqrt(1+4*t_k**2)) / 2
        beta_kp = (t_k - 1)/t_kp
        dx = x_kp - x_k
        y_kp = x_kp + beta_kp*dx
        restart = np.sum((y_k - x_kp)*(dx))
        if restart > 0 and do_restarting is True:
            print("restarting")
            t_k = 1
        else: 
            t_k = t_kp
            
        x_k = x_kp
#        t_k = t_kp
        y_k = y_kp
        f_out.append(f)

    return y_k, f_out

def linear_gradient(x, A, A_adj, b):

    r = A(x) - b
    return A_adj(r), np.linalg.norm(r.ravel())**2
    
    
def soft_wavelets_skimage(x, tau, pos = False, shift = False, ycbcr = True):
    xshift = int(np.round(np.random.rand(1)))
    yshift = int(np.round(np.random.rand(1)))
    #print("xshift:")
    #print(xshift)
    x_in = x
    if shift: x = np.roll(x, (yshift, xshift), axis=(-2,-1))
    x = np.moveaxis(denoise_wavelet(np.moveaxis(x,0,-1), multichannel=True, convert2ycbcr=ycbcr,
                                 mode='soft',sigma=tau), -1,0)
    if shift: x = np.roll(x, (-yshift, -xshift), axis=(-2,-1))
    
    if pos:
#        x_out = .5*x_d + .5*np.maximum(x,-10000)
        x_out = .5*x + .5*np.maximum(x_in,0)
    else:
        x_out = x
    return x_out

def soft_wavelets_skimage_cycle(x, tau, shift=0, pos = False, ycbcr = True):
    xshift = shift % 2
    yshift = (shift//2) % 2

    x_in = x
    x = np.roll(x, (yshift, xshift), axis=(-2,-1))
    x = np.moveaxis(denoise_wavelet(np.moveaxis(x,0,-1), multichannel=True, convert2ycbcr=ycbcr,
                                 mode='soft',sigma=tau), -1,0)
    x = np.roll(x, (-yshift, -xshift), axis=(-2,-1))
    
    if pos:
#        x_out = .5*x_d + .5*np.maximum(x,-10000)
        x_out = .5*x + .5*np.maximum(x_in,0)
    else:
        x_out = x
    return x_out

def prox4learning(x, kernel_list, bias_list):
    for n in range(np.shape(kernel_list)[0]):
        y = signal.convolve2d(x, kernel_list[n],mode='same')
        x = np.maximum(y + bias_list[n],0)
        
    return x

def prox4learningRGB(x, kernel_list, bias_list):
    n_layers = np.shape(kernel_list)[0]

    for n in range(n_layers):
        y = np.array([signal.convolve2d(x[m], kernel_list[n][m,:,:],mode='same') for m in range(3)])
        x = np.maximum(y + bias_list[n],0)
        
    return x
