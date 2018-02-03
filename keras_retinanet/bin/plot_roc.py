import os, sys
import matplotlib.pyplot as plt
import pickle
import math

metrics = pickle.load(open(sys.argv[1], 'r'))

average_precisions = metrics['average_precisions']
precisions = metrics['precisions']
recalls = metrics['recalls']

# print evaluation
grid = int(math.ceil(math.sqrt(len(average_precisions.keys()))))
fig, ax = plt.subplots(nrows=grid, ncols=grid)
for label, average_precision in average_precisions.items():
    print(label, '{:.4f}'.format(average_precision))
    ax.step(recalls[label], precisions[label], color='b', alpha=0.2,
                 where='post')
    ax.fill_between(recalls[label], precisions[label], step='post', alpha=0.2,
                     color='b')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('cls {0:} : AUC={1:0.2f}'.format(label, average_precision))
plt.suptitle("Mean average precision : {:0.2f}".format(sum(average_precisions.values()) / len(average_precisions)))
fig.tight_layout()
plt.savefig(sys.argv[2])
plt.show()
