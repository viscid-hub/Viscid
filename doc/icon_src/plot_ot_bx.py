import matplotlib.pyplot as plt
import seaborn as sns
import viscid
from viscid.plot import vpyplot as vlt

f = viscid.load_file('./otico_001.3d.xdmf')

mymap = sns.diverging_palette(28, 240, s=95, l=50, as_cmap=True)

figure = plt.figure(figsize=(14, 10))
g = f.get_grid(time=12)
vlt.plot(g['bx']['z=0'], cmap=mymap, style='contourf', levels=256)
vlt.savefig('OT_bx.png')
plt.show()
