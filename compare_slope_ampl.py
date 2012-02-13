from calc_adj import calc_adj, load_ifuncs, load_displ_legendre

if 'ifuncs' not in globals():
    ifuncs = load_ifuncs('RY', 'p')
    ifuncsx = load_ifuncs('X', 'p')

if 'displ' not in globals():
    displx = load_displ_legendre(ifuncsx, 8, 4, rms=5.0)
    displry = (displx)
    # OR displ = load_displ_grav('RY', 'p')

if 'clip' not in globals():
    clip = 20

if 'n_ss' not in globals():
    n_ss = 5

# coeffs, adj, M_2d = calc_adj(ifuncs, displ, n_ss, clip)
