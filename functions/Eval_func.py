from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Get metrics
def metrics_error(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    return mae, rmse

# Get metrics from data and return table
def metrics_error_table(data):
    mae_nn, rmse_nn = metrics_error(data['totalFare'], data['nn_predicted_totalFare'])
    mae_xgb, rmse_xgb = metrics_error(data['totalFare'], data['xgb_predicted_totalFare'])
    mae_lgbm, rmse_lgbm = metrics_error(data['totalFare'], data['lgbm_predicted_totalFare'])
    return pd.DataFrame({
        'Model': ['NN', 'XGB', 'LGBM'],
        'MAE': [mae_nn, mae_xgb, mae_lgbm],
        'RMSE': [rmse_nn, rmse_xgb, rmse_lgbm]
    })

def set_custom_theme():
    sns.set_style('whitegrid')  # Set the style to whitegrid
    sns.set_palette(sns.color_palette(['#a8e6cf', '#dcedc1', '#ffd3b6', '#ffaaa5', '#ff8b94']))  # Set the custom palette
    # Manually set font sizes
    plt.rc('axes', titlesize=14)     # Font size for axes titles
    plt.rc('axes', labelsize=12)     # Font size for x and y labels
    plt.rc('xtick', labelsize=10)    # Font size for x tick labels
    plt.rc('ytick', labelsize=10)    # Font size for y tick labels
    plt.rc('legend', fontsize=12)    # Font size for legend
    plt.rc('font', size=12)          # General font size
    

# Apply the custom theme
set_custom_theme()

# Plot true vs predicted scatter using sns
def plot_true_vs_pred(y_true, y_pred, title='Actual vs Predicted',y_pred2=None):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7)
    if y_pred2 is not None:
        # Add second scatter plot colour coded using last colour in palette
        sns.scatterplot(x=y_true, y=y_pred2, alpha=0.7, color=sns.color_palette()[-1])
        plt.legend([ 'NN','XGB'])
    plt.plot([0, max(y_true)], [0, max(y_true)], color='#d3d3d3', linewidth=1, linestyle='--')  # Light grey dashed line
    plt.title(title, pad=30)  # Add padding to the title
    plt.ylim(0, 5000 )
    plt.xlim(0, 5000 )
    plt.xlabel('Actual', labelpad=15)  # Add padding to the x-axis label
    plt.ylabel('Predicted', labelpad=15)  # Add padding to the y-axis label
    
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Add white space border

# Plot Airport Route Facet Plot
def route_facet_plot(data, col_facet, x,model_col, title, label=None):
    g = sns.FacetGrid(data, col=col_facet, col_wrap=3, height=4, aspect=1)
    print(np.arange(min(data[x]), max(data[x]),1))
    if label is not None:
        g.map_dataframe(sns.barplot, x=x, y='totalFare', label='Actual',alpha=0.7, order=np.arange(min(data[x]), max(data[x])+1,1))
    else:
        g.map_dataframe(sns.barplot, x=x, y='totalFare', label='Actual',alpha=0.7)
    
    g.map_dataframe(
        sns.barplot,
        x=x,
        y=model_col,
        label="Predicted",
        color=sns.color_palette()[-3],
        alpha=0.5,
        errorbar=None,
        
        linewidth=2.5, 
        linestyle= ":",
        edgecolor="orange",
       
        
    )
    g.set_titles(col_template='{col_name}')

    g.set_axis_labels(x, 'Avg Fare') # Set the axis labels
    g.add_legend()  # Add the legend

    # set tickmarks under every graph 1 = ULC, 2=Budget, 3=Regional, 4=Full Service
    if label is not None:
        g.set_xticklabels(label)
    g.tick_params(axis='x', labelbottom=True, )
    # set title
    g.fig.suptitle(title, y=1.05, fontsize=16)    
