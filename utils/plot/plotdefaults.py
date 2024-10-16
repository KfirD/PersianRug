import matplotlib.pyplot as plt
import matplotlib

plt.rcParams.update({"text.usetex": True,'text.latex.preamble': r'\usepackage{amsfonts}' + '\n' + r'\usepackage{amsmath}'})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rc('font', **{'size': 22})
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
