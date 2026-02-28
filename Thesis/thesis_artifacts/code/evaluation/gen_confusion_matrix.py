import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

cm = np.array([
    [30558,     0,   117,    48],
    [    0,   524,     0,     0],
    [  181,     0,  9810,   645],
    [   70,     0,   535, 10015],
], dtype=float)

row_sums = cm.sum(axis=1, keepdims=True)
cm_norm = cm / row_sums

classes = ['Asphalt', 'Cobblestone', 'Gravel', 'Sand']

fig, ax = plt.subplots(figsize=(6.5, 5.4))

cmap = plt.cm.Blues
im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Anteil pro wahrer Klasse', fontsize=10)
cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

tick_marks = np.arange(len(classes))
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(classes, fontsize=11)
ax.set_yticklabels(classes, fontsize=11)
ax.set_xlabel('Vorhergesagte Klasse', fontsize=12, labelpad=8)
ax.set_ylabel('Wahre Klasse', fontsize=12, labelpad=8)

thresh = 0.5
for i in range(len(classes)):
    for j in range(len(classes)):
        val = cm_norm[i, j]
        color = 'white' if val > thresh else 'black'
        pct = f'{val:.1%}'
        n   = f'(n\u202f=\u202f{int(cm[i,j]):,})'
        ax.text(j, i - 0.12, pct, ha='center', va='center',
                fontsize=11, color=color, fontweight='bold')
        ax.text(j, i + 0.25, n,   ha='center', va='center',
                fontsize=7.5, color=color)

ax.set_title(
    'Normalisierte Konfusionsmatrix \u2013 MobileNetV3-Small Baseline\n'
    '(Testset, 52\u202f503 Patches)',
    fontsize=11, pad=12
)

plt.tight_layout()

out_dir = '/Users/nikitamasch/Downloads/merged/thesisModels/content/attachments'
os.makedirs(out_dir, exist_ok=True)

pdf_path = os.path.join(out_dir, 'texture_confusion_matrix_normalized_NEW.pdf')
png_path = os.path.join(out_dir, 'texture_confusion_matrix_normalized_NEW.png')

plt.savefig(pdf_path, bbox_inches='tight', dpi=200)
plt.savefig(png_path, bbox_inches='tight', dpi=200)
print('Saved PDF:', pdf_path)
print('Saved PNG:', png_path)
