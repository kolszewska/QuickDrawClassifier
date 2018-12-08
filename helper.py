import matplotlib.pyplot as plt
import numpy as np


def view_classification_result(img, ps):
    """Function for viewing an image and it's predicted classes."""
    ps = ps.data.numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(['Bear', 'Bee', 'Camel', 'Cat', 'Cow', 'Crab', 'Crocodile', 'Dog', 'Dolphin', 'Duck'])
    ax2.set_title('Class probabilities')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
