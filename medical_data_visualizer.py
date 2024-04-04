import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
df['overweight'] = np.where(df['weight'] / (df['height']/ 100)**2 > 25, 1 , 0)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.

df['cholesterol'] = np.where(df['cholesterol'] == 1, 0, 1)
df['gluc'] = np.where(df['gluc'] == 1, 0, 1)

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df, id_vars = 'cardio', value_vars= ['cholesterol','gluc','smoke','alco','active','overweight']).copy()


    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = df_cat.groupby(['cardio','variable','value']).size().reset_index(name = 'counts')
    

    # Draw the catplot with 'sns.catplot()'
    sns.catplot(
    data = df_cat,
    x = 'variable',
    y = 'counts',
    col = 'cardio',
    kind = 'bar',
    hue = 'value'
    )

    # Get the figure for the output
    fig = sns.catplot(
    data = df_cat,
    x = 'variable',
    y = 'counts',
    col = 'cardio',
    kind = 'bar',
    hue = 'value'
    )

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
          (df['height'] >= df['height'].quantile(0.025)) &
          (df['height'] <= df['height'].quantile(0.975)) &
          (df['weight'] >= df['weight'].quantile(0.025)) &
          (df['weight'] <= df['weight'].quantile(0.975))
          ].copy()

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize = (10,10))
    corr_masked = np.ma.array(corr, mask = mask)
    ax.set_xticks(ticks = np.arange(0,14), labels = corr.columns.to_list(), rotation = 'vertical')
    ax.set_yticks(ticks = np.arange(0,14), labels = corr.columns.to_list())
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.colorbar(ax.imshow(corr_masked), shrink = 0.5)
    plt.show()

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(
        data = corr,
        mask = mask,
        annot = True,
        fmt = '.1f'
    )
    plt.show()


    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
