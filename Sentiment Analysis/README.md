# Sentiment Analysis - NLTK

# Introduction

Sentiment Analysis is used to determine the emotional tone behind words to gain understanding of the attitudes, opinions and emotions expressed within an online mention. It is also known as opinion mining, deriving the opinion or attitude of a speaker.

NLTK (Natural Language Toolkit) is a robust tool for both complex and simple text analysis. NLTK contains many modules and algorithms, such as sentiment analysis. 

The goal for this project is to use sentiment analysis with NLTK to quantify the comments from Yelp for the targeted restaurant.

# Data Summary

I uesd Beautiful Soup package in Python to grab related data from the restaurant’s Yelp page, including author, ratingValue, datePublished, description. And we had 222 instances at all. Later, I will do the sentiment analysis with the descriptions from each customers.  

# Algorithm

Then I continued to use the comments gathered from our previous step, and used NLTK to do sentiment analysis among comments data.

Firstly, I used tokenize to break the comment into sentences for each customers.

Secondly, I used the polarity_scores() method to find the positive, negative, neutral and compound scores of each sentence. 

Thirdly, we wanted to find the average compound sentiment for each review. To do this, we could find the sum the compound score as the numerator and the count of sentences as the denominator.

Finally, we have a dataset we can place in a dataframe, store in a csv, or use in further analysis. The datafile have 5 columns, including author, ratingValue, datePublished, description and average compound sentiment score, with 222 instances.

# Result

The overall average sentiment was 0.354171, meaning there was a positive sentiment towards the restaurant.
