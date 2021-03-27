import feedparser
import csv
#news_rss_url = "https://answers.yahoo.com/rss/allq"
#info = feedparser.parse(news_rss_url)

with open('C:\\Users\\Admin\\Desktop\\ProjectLDA\\answers.txt','a') as fd:
#with open('C:\\Users\\Admin\\Desktop\\ProjectLDA\\test.txt','a') as fd:
    writer = csv.writer(fd, delimiter=',', lineterminator = '\n')
    #for entry in info.entries:
        #writer.writerow([entry.title])
