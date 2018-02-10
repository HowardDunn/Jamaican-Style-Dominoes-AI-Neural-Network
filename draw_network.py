import matplotlib.pyplot as plt 
from draw_neural_net import draw_neural_net

fig = plt.figure(figsize=(50, 50))
ax = fig.gca()
ax.axis('off')
draw_neural_net(ax, .1, .9, .1, .9, [14, 56, 28, 1])
fig.savefig('nn.png')