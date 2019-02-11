# import the package
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import data
from urllib.request import urlopen
from bs4 import BeautifulSoup #get the reviews
import pandas as pd  #dataframe and csv
import matplotlib.pyplot as plt 

splitter = data.load('tokenizers/punkt/english.pickle')
sid = SentimentIntensityAnalyzer()

# the restaurant name: Jazz Standard
# the yelp page of the restaurant: https://www.yelp.com/biz/jazz-standard-new-york

#Gather comments from all yelp pages 
allComments = [] 
start = 1 
while start < 13: # the overall page we have right now is 12
	urlString = r'https://www.yelp.com/biz/jazz-standard-new-york?start= '
	urlString =urlString[:-1] + str((start-1)*20)     
	url = urlopen(urlString) 
	
	# using beautiful soup to read the link and find the infomation
	soup = BeautifulSoup(url, 'html.parser')     
	reviewList = soup.find_all(itemprop='review')  
       
        # find infomation we need
	for review in reviewList:         
		author = review.find(itemprop='author')['content']         
		stars = review.find(itemprop='ratingValue')['content']         
		date = review.find(itemprop='datePublished')['content']         
		description = review.find(itemprop='description').get_text().replace('\n', '').replace(',', '')         
		commentTuple = (author, stars, date, description)         
		allComments.append(commentTuple)     
	start += 1


# first comments
allComments[0]	
# last comments
allComments[-1]	
# how many comments we have right now?
len(allComments)   # 222 comments

	
# find the average compound sentiment for each review.  
commentsAndSentiment = []  
for entry in allComments:
    # tokenize the comment to break it into sentences: 
    sentences = splitter.tokenize(entry[3]) 
    sentenceCount = len(sentences) 
    sentimentTotal = 0
	
    # use the polarity_scores() method to find the positive, negative, and compound scores of each sentence.  
    for s in sentences: 
        ss = sid.polarity_scores(s) 
        #print(s, ss) 
								
       # find the average compound sentiment for each review.          
       # the sum the compound score as the numerator
       # the count of sentences as the denominator
        compound = ss['compound'] 
        positive = ss['pos'] 
        neutral = ss['neu'] 
        negative = ss['neg'] 
        sentimentTotal += compound 
        #print(s, compound, positive, neutral, negative) 
    sentimentAverage = sentimentTotal/sentenceCount 
    #print(entry[0], entry[1], entry[2], entry[3], sentimentAverage) 
    sentimentTuple = (entry[0], entry[1], entry[2], entry[3], sentimentAverage) 
    commentsAndSentiment.append(sentimentTuple) 
	
# store in a csv
# create dataframe  
commentFrame = pd.DataFrame(commentsAndSentiment, columns=['author', 'stars', 'date', 'comment', 'sentiment']) 

# output to file (define the path and file name) 
commentFrame.to_csv(r'yelp_reviews.csv') 

#cast date column to datetime datatype and set it as the index 
commentFrame['date'] = pd.to_datetime(commentFrame['date'], errors='coerce')  
commentFrame.set_index('date', inplace=True)
commentFrame

#cast sentiment column as float datatype 
commentFrame['sentiment'] = commentFrame['sentiment'].astype(float) 
commentFrame['stars'] = commentFrame['stars'].astype(float) 

# overall average sentiment 
commentFrameAvg = commentFrame[['sentiment']].mean() 
commentFrameAvg       # 0.354171
commentFrameStarAvg = commentFrame[['stars']].mean()  
commentFrameStarAvg   # 4.126126

#plot average sentiment by month 
#aggregate the data by the average per month 
commentFrameByMonth = commentFrame.resample('m')[['sentiment']].mean() 
#drop the months that did not have a comment 
commentFrameByMonth = commentFrameByMonth.dropna() 
commentFrameByMonth

#plot the data using matplotlib -be sure to import pyplot at the top of the script 
x = commentFrameByMonth.index 
y = commentFrameByMonth['sentiment'].values 
 
plt.plot(x, y) 
plt.title('Sentiment Analysis By Month')
plt.ylabel('Sentiment Score')
plt.xlabel('Month')
plt.legend()
plt.show()

#aggregate the data by the average per year 
commentFrameByYear = commentFrame.resample('y')[['sentiment']].mean() 
commentFrameByYear = commentFrameByYear.dropna() 
commentFrameByYear

a = commentFrameByYear.index 
b = commentFrameByYear['sentiment'].values 

plt.plot(a, b) 
plt.title('Sentiment Analysis By Year')
plt.ylabel('Sentiment Score')
plt.xlabel('Year')
plt.legend()
plt.show()
