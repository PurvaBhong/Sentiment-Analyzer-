import torch
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt


from transformers import pipeline
model_path = ("distilbert/distilbert-base-uncased-finetuned-sst-2-english")

#path used for local testing
#analyzer = pipeline("text-classification", model=model_path)

# path used to deploy project on huggingface spaces
analyzer = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# print(analyzer(["I love using transformers library!", "chatgpt pro version is quite expensive!"]))

def sentiment_analyzer(review):
    sentiment = analyzer(review)
    return sentiment[0]['label']

def sentiment_bar_chart(df):
    sentiment_counts = df['Sentiment'].value_counts()
    
    #Create a bar chart
    fig, ax = plt.subplots()
    sentiment_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax, color=['blue', 'orange'])
    ax.set_title('Review Sentiment Counts')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    #ax.set_xticklabels(['Positive', 'Negative'], rotation=0)
    
    #Return the figure return fig
    return fig

def read_reviews_and_analyze_semtiment(file_object):
    df = pd.read_excel(file_object)   # Load the Excel file into a DataFrame


    if 'Reviews' not in df.columns:
        raise ValueError("Excel file must contain a 'Review' column.")
    
    df['Sentiment'] = df['Reviews'].apply(sentiment_analyzer)
    chart_object = sentiment_bar_chart(df)
    return df, chart_object

#results = read_reviews_and_analyze_semtiment("sample_data.xlsx")
#print(results)

demo = gr.Interface( fn= read_reviews_and_analyze_semtiment,
                    inputs=[gr.File(file_types=[".xlsx"], label="Upload your review file")],
                    outputs=[gr.Dataframe(label="Sentiment"), gr.Plot(label="Sentiment Analysis Bar Chart")], 
                    title="Sentiment Analyzer",
                    description="Analyze the sentiment of your text using a pre-trained DistilBERT model.")

demo.launch()


