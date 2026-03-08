import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast

def load_data(file_path):
    """Load the CSV file."""
    return pd.read_csv(file_path)

def process_data(df):
    """Process the data, e.g., convert string list into actual lists."""
    df['block_size_list'] = df['block_size_list'].apply(ast.literal_eval)
    return df

def plot_violin(df):
    """Plot a violin plot for performance metrics."""
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df[['masked_WER', 'WER_bigvgan', 'WER_hifi_gan']])
    plt.title('Violin Plot of WER Metrics')
    plt.ylabel('WER')
    plt.show()

def plot_histogram(df):
    """Plot a histogram for the STOI metric."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df['STOI_bigvgan'], kde=True, color='blue', label='STOI_bigvgan', alpha=0.6)
    sns.histplot(df['STOI_hifi_gan'], kde=True, color='green', label='STOI_hifi_gan', alpha=0.6)
    plt.title('Histogram of STOI Metrics')
    plt.xlabel('STOI')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def plot_box(df):
    """Plot a box plot for selected performance metrics."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[['LSD_init_bigvgan', 'LSD_init_hifi_gan']])
    plt.title('Box Plot of LSD Metrics')
    plt.ylabel('LSD')
    plt.show()

def plot_scatter(df):
    """Plot a scatter plot for the relationship between WER and PLCMOS."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['masked_WER'], y=df['plcmos_pred_bigvgan'])
    plt.title('Scatter Plot of Masked WER vs PLCMOS Prediction (BigVGAN)')
    plt.xlabel('Masked WER')
    plt.ylabel('PLCMOS Prediction (BigVGAN)')
    plt.show()

def main(file_path):
    df = load_data(file_path)
    df = process_data(df)
    
    # Generate and display the plots
    plot_violin(df)
    plot_histogram(df)
    plot_box(df)
    plot_scatter(df)

if __name__ == "__main__":
    # Replace with the path to your CSV file
    file_path = '/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/Unet_Anechoic_LibSp_wavlm-conditional_w-masked-pix=0.8/unet_dim64_dim_mults1_2_4_T400_betaT0.02/as-train-gap_cp=532000_mel_text=True_withoutLM/w1=2_w2=0.8_asr_start=320_mask=True/metric_results_and_samples_info.csv'
    main(file_path)
