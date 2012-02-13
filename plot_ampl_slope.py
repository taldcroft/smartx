if 0:
    ifuncs = load_ifuncs('RY', 'p')
    ifuncsx = load_ifuncs('X', 'p')

figure(2, figsize=(6,6))
clf()
x0 = 214
y0 = 430
sz = 50
xr = slice(x0 - sz, x0 + sz)
yr = slice(y0 - sz, y0 + sz)
if10 = ifuncs[10, 10]
ifx10 = ifuncsx[10, 10]

subplot(2, 2, 1)
ax = gca()
ax.axison = False
imshow(ifx10[xr, yr])
title('Amplitude')

subplot(2, 2, 2)
plot(ifx10[xr, y0])
title('Amplitude')
grid()

subplot(2, 2, 3)
ax = gca()
ax.axison = False
imshow(if10[xr, yr])
title('Slope')

subplot(2, 2, 4)
plot(if10[xr, y0])
title('Slope')
grid()

tight_layout()

savefig('ifuncs_ampl_slope.png')
