import glob
import matplotlib.pyplot as plt
import asciitable

drive_ref = 'median'
lines = []
for filename in glob.glob('drive_noise-cygob2-*.dat'):
    lines.extend(open(filename, 'r').readlines())

vals = asciitable.read(lines, Reader=asciitable.NoHeader, guess=False,
                       names=['noise', 'hpd', 'rmsd'])

drive_noise = vals['noise'] * 5.0  # Ugh, calc_drive_noise was run with drive_ref=5.0
plt.figure(1, figsize=(6, 4))
plt.clf()
plt.plot(drive_noise, vals['rmsd'], '.b', mec='b')
plt.xlabel('Drive noise (where 1.0 == 100ppm strain)')

plt.title('RMSD vs. drive noise')
plt.ylabel('RMSD (arcsec)')
plt.ylim(0, None)
#plt.xlim(-0.02, 0.22)
plt.tight_layout()
plt.grid()
plt.savefig('rmsd_drive_noise_bias.png')


plt.figure(2, figsize=(6, 4))
plt.clf()
plt.plot(drive_noise, vals['hpd'], '.b', mec='b')
plt.xlabel('Drive noise (where 1.0 == 100ppm strain)')

# plt.title('HPD dependence on drive noise')
plt.ylabel('HPD (arcsec)')
plt.ylim(0, None)
#plt.xlim(-0.02, 0.22)
plt.tight_layout()
plt.title('HPD vs. drive noise')
plt.grid()
plt.savefig('hpd_drive_noise_bias.png')
