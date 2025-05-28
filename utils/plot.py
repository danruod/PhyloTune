from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd

abbr2name = {'attn': 'Average attention', 
             'hetero': 'Heterozygosity', 
             'fst': 'FST', 
             'dxy': 'DXY', 
             'sub_rate': 'Substitution rate'
             }    

def plot_confusion_matrix(figsize, labels, preds, labels_id, labels_name, save_path=None):
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    
    cm = confusion_matrix(labels, preds, labels=labels_id)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_name)

    disp.plot(ax=ax, xticks_rotation='vertical')
    
    if save_path is not None:
        plt.savefig(save_path)
    
    return


def plot_attn_analysis(attn_scores, results_dict, pearson_metric, save_dir, marker, heatmap_attn=True):
    
    max_score = attn_scores.max()
    min_score = attn_scores.min()
    print(f"--- max/min attention score: {max_score}/{min_score}")
    
    # sns.set(font_scale=8)
    sns.set_theme(style="white", palette="tab10")
    # color_hex = sns.color_palette("Set3").as_hex()
    
    plt_num = len(results_dict)
    height_ratios = [1] * plt_num
    
    if heatmap_attn:
        plt_num += 2
        height_ratios = [2] + height_ratios
        
    fig = plt.figure(figsize=(10, plt_num*3.1), dpi=300)    
    gs = gridspec.GridSpec(len(height_ratios), 1, height_ratios=height_ratios, hspace=0.3)
    
    i = 0
    if heatmap_attn:
        ax = fig.add_subplot(gs[i])
        yticklabels = ['0'] + [''] * 199 + ['200'] + [''] * 199 + ['400'] + [''] * 199 + ['600'] + [''] * 199 + ['800'] + [''] * 199
        xticklabels = []
        for j in range(0, attn_scores.shape[1], 200):
            xticklabels = xticklabels + [j] + [''] * min(199, attn_scores.shape[1]-j-1)
        sns.heatmap(attn_scores, cmap='YlGnBu', vmin=min_score, vmax=max_score, xticklabels=xticklabels, yticklabels=yticklabels)
        ax.set_title(f'Attention for {marker}')
        i = 1
    
    for j, (key, value) in enumerate(results_dict.items()):
        ax = fig.add_subplot(gs[i+j])
        
        if key.split("-")[0] in ['attn', 'fst', 'dxy']:
            sns.lineplot(value, linewidth=1, color='black')
            # sns.lineplot(-np.log10(1-value), linewidth=1, color='black')
        # elif args.plot_type == 'bar':
        #     sns.barplot(x='index', y='value', data=pd.DataFrame(value, columns=['value']).reset_index(), color='black')
        else:
            try:
                sns.regplot(pd.DataFrame(value, columns=['value']).reset_index(), x='index', y='value', order=100, scatter_kws={'s': 1, 'color': 'black'}, 
                            line_kws={'linewidth': 1, 'color':'black'})
            except:
                plt.cla()
                sns.regplot(pd.DataFrame(value, columns=['value']).reset_index(), x='index', y='value', order=50, scatter_kws={'s': 1, 'color': 'black'}, 
                            line_kws={'linewidth': 1, 'color': 'black'})

        if key.split("-")[0] == 'attn':
            ax.set_title(f'Average attention')
        else:
            try:
                # get pearson_metric value based on key1 and key2
                pearson_value = pearson_metric[(pearson_metric['key1'] == 'attn') & (pearson_metric['key2'] == key)]
            except:
                pearson_value = pearson_metric[(pearson_metric['key1'] == key) & (pearson_metric['key2'] == 'attn')]
            
            # 'fst': r"F_{ST}", 'dxy': r"\textit{D}_{XY}", 
            title = f'(r: {pearson_value["coeff"].values[0]:.2f}, p: {pearson_value["pvalue"].values[0]:.2f})'
        
            if key.split("-")[0] == 'fst':
                ax.set_title(r"F$_{ST}$ " + title)
            elif key.split("-")[0] == 'dxy':
                ax.set_title(r"D$_{XY}$ " + title)
            else:
                ax.set_title(f'{abbr2name[key.split("-")[0]]} {title}')
            
        plt.xlabel('')
        plt.ylabel('')
        plt.ylim(ymin=0)
        plt.ylim(ymax=min(value.max(), 1.0))
          
    plt.savefig(f'{save_dir}/{marker}.png')

    # plt.clf()
    plt.close()
