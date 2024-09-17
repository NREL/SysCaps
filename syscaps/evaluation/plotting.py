
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as dates


def plot_energyplus(timestamps, ground_truth, prediction, range, model_name, title):
    """
    Args:
        timestamps: (n,1) numpy array
        ground_truth: (n,1)
        prediction: (n,1)
        range: (start, end]
    """
    #opacity = 0.6
    linestyles = ['--', '-']
    start, end = range

    plt.plot(timestamps[start:end], ground_truth[start:end:], color='black',  linewidth=3, linestyle=linestyles[0], label='Ground truth')    
    
    plt.plot(timestamps[start:end], prediction[start:end], color='blue', linewidth=3, label=model_name, linestyle=linestyles[1])

    plt.legend(fontsize=18)
    #ax = plt.gca()
    #ax.xaxis.set_major_formatter(dates.DateFormatter('%H'))  # hours and minutes
    # Rotate date labels automatically
    #plt.gcf().autofmt_xdate()
    # Reduce the size of the xlabels
        
    plt.title(title, fontsize=22)
    plt.ylabel('kWh', fontsize=22)
    plt.xlabel('Hour of day', fontsize=22)
    #plt.xticks(rotation=45, fontsize=14)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)