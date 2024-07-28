# src/visualize.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_data(df):
    # Correlation Matrix
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()

    # Pair Plot
    sns.pairplot(df, hue='PlacementStatus', vars=['CGPA', 'SoftSkillsRating', 'AptitudeTestScore', 'SSC_Marks', 'HSC_Marks'])
    plt.show()

    # Distribution of CGPA
    plt.figure(figsize=(10, 6))
    sns.histplot(df['CGPA'], kde=True, bins=30)
    plt.title('Distribution of CGPA')
    plt.xlabel('CGPA')
    plt.ylabel('Frequency')
    plt.show()

    # Box Plot of SSC and HSC Marks
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[['SSC_Marks', 'HSC_Marks']])
    plt.title('Box Plot of SSC and HSC Marks')
    plt.show()
