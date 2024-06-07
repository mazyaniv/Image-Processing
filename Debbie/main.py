import spectral
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.integrate import cumtrapz
from numpy import linalg as LA

#############################################################################Functions
def visual(data,name): #Visualize the cube
    # Choose four different bands to display
    data = (255 / np.max(data.flatten())) * data
    bands = [10, 20, 30, 40]
    # Create a figure with subplots to display the band images
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    # Display each band image in a subplot
    for band, ax in zip(bands, axes.flat):
        band_image = data[:, :, band]
        ax.imshow(band_image, cmap='gray')
        ax.set_title('Band {}'.format(band))
    plt.suptitle("{}".format(name))
    plt.show()
    
def plot_pixel_spectra(cube, row, col): #Visual the target spectral
    pixel_spectra = cube[row, col, :]
    fig, ax = plt.subplots()
    ax.plot(pixel_spectra)
    ax.set_title('Target Spectra at Row={}, Col={}'.format(row, col))
    ax.set_xlabel('Band Number')
    ax.set_ylabel('Reflectance')
    plt.show()
#############################################################################Using .dat & .hdr file
img = spectral.open_image('bimodal.hdr')
data = np.fromfile('bimodal.dat', dtype=img.dtype)
cube = np.reshape(data, (img.shape[0], img.shape[1], img.shape[2]), order="F")

plot_pixel_spectra(cube, 2, 4) #target
visual(cube,"Data")
############################################################################# M,phi
t = cube[2, 4, :]
p = 0.01
M = np.zeros(np.shape(cube))
for i in range(M.shape[-1]):
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    padded_matrix = np.pad(cube[:,:,i], (1, 1), mode='constant', constant_values=0)
    neighbor_count = convolve2d(np.ones_like(cube[:,:,i]), np.ones((3, 3)), mode='same')
    neighbor_sum = convolve2d(padded_matrix, kernel, mode='valid')
    M[:,:,i] = neighbor_sum/(neighbor_count-1)
X_MINUS_M = np.subtract(cube,M)
arg_cov = X_MINUS_M.reshape(img.shape[0]*img.shape[1],img.shape[2])
phi = np.cov(arg_cov.transpose())  #Transpose- Python vs. Matlab
t_inv_phi = np.dot(t,LA.inv(phi))

visual(M,"M")
#############################################################################Histogrm
MF_NT = np.zeros((img.shape[0],img.shape[1]))
MF_WT = np.zeros((img.shape[0],img.shape[1]))
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        MF_NT[i,j] = np.dot(t_inv_phi,(cube[i,j,:]-M[i,j,:]))
        MF_WT[i, j] = np.dot(t_inv_phi,(cube[i,j,:]-M[i,j,:]+p*t))

for i in range(4,20,5): #Checking
    print("X_minus_M_NT {},{}:".format(i,i),cube[i,i,0]-M[i,i,0])
    print("X_minus_M_WT {},{}:".format(i,i),cube[i,i,0]-M[i,i,0]+p*t[0])
print("--------------------")
for i in range(4,20,5): #Checking
    print("MF_NT {},{}:".format(i,i), MF_NT[i,i])
    print("MF_WT {},{}:".format(i,i), MF_WT[i,i])
###############################################Plot
NT_val, NT_bins = np.histogram(MF_NT, bins=100)
WT_val, WT_bins = np.histogram(MF_WT, bins=100)
fig, ax = plt.subplots()
ax.plot(NT_bins[:-1], NT_val, label='No target')
ax.plot(WT_bins[:-1], WT_val, label='With target')
ax.set(title='Original Histogram', xlabel='Bins', ylabel='Frequency')
ax.legend()
plt.ylim(0,120)
plt.show()
#############################################################################Inverse_CDF
Pd = 1-(np.cumsum(WT_val)/np.sum(WT_val)) #1-((cumtrapz(WT_val, WT_bins[:-1]))/(cumtrapz(WT_val, WT_bins[:-1])[-1]))
Pfa = 1-(np.cumsum(NT_val)/np.sum(NT_val)) #1-((cumtrapz(NT_val, NT_bins[:-1]))/(cumtrapz(NT_val, NT_bins[:-1])[-1]))

plt.plot(WT_bins[:-1], Pd,label="Pd")
plt.plot(NT_bins[:-1], Pfa,label="Pfa")
plt.xlabel('bins')
plt.ylabel('Cumulative Probability')
plt.title('Inverse Cumulative Probability distribution')
plt.legend()
plt.show()
#############################################################################ROC curve
th = 0.1
index = max(100-np.argmax(WT_bins > NT_bins[-1]), np.argmax(NT_bins > WT_bins[0]))
Pd_new = np.pad(Pd, (index,0), mode='constant', constant_values=(1,1))
Pfa_new = np.pad(Pfa, (0,index), mode='constant', constant_values=(0,0))

plt.plot(Pfa_new, Pd_new)
plt.plot([0, th], [0, th], '--')
plt.title('ROC Curve')
plt.xlim(0, th)
plt.show()
#############################################################################Performance
Pd_new = np.flip(Pd_new)
Pfa_new = np.flip(Pfa_new)

def area(th):
    cond = np.where(Pfa_new > th)[0][0]
    Pd_mask = Pd_new.copy()
    Pd_mask[cond:] = 0
    A = cumtrapz(Pd_mask, Pfa_new)
    return (A-0.5*pow(th,2))/(th-0.5*pow(th,2))

print("Performance, using 'area test':")
print("th= 0.001:", area(0.001)[-1])
print("th= 0.01:", area(0.01)[-1])
print("th= 0.1:", area(0.1)[-1])
