import pandas as pd
data=pd.read_csv("D:\\ml.py\\datasets\\imdb.csv")

x=data['review']
y=data['sentiment']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

from sklearn.feature_extraction.text import CountVectorizer
vc=CountVectorizer(stop_words="english")
x_train=vc.fit_transform(x_train)
x_test=vc.transform(x_test)

from sklearn.naive_bayes import MultinomialNB
mb=MultinomialNB()
mb.fit(x_train,y_train)
ypred=mb.predict(x_test)

from sklearn.metrics import accuracy_score
ac=accuracy_score(ypred,y_test)
print(ac*100)

a=["Bob's is a Good boy"]
a=vc.transform(a)
predict=mb.predict(a)

if predict == "positive":
    print("positive")
else:
    print("negative")
