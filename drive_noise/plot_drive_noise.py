import glob
import matplotlib.pyplot as plt
import asciitable

drive_ref = 'median'
lines = []
for filename in glob.glob('drive_noise-*.dat'):
    lines.extend(open(filename, 'r').readlines())

vals = asciitable.read(lines, Reader=asciitable.NoHeader, guess=False,
                       names=['noise', 'hpd', 'rmsd', 'displ_std',
                              'resid_std'])

plt.figure(1, figsize=(5, 3.5))
plt.clf()
plt.plot(vals['noise'], vals['hpd'] / vals['hpd'].min(), ',b', mec='b')
plt.xlabel('Drive noise relative to {} drive level'
           .format(drive_ref))
# plt.title('HPD dependence on drive noise')
plt.ylabel('HPD relative to zero-noise case')
plt.ylim(0, None)
plt.xlim(-0.02, 0.22)
plt.tight_layout()
plt.grid()
plt.savefig('drive_noise.png')
