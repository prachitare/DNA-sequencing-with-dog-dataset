import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#importing the dattaset

db = pd.read_table('dog_data.txt')

#print(db.head())

# k-mer ki help se poore dataset ko divide karna hai 6 6 workds mai aur uuse join karna hai to make it easy. k-mer basicaly belongs to hexamer yaani sequence ko 6 mai tod dena so that we can classify it more better

def splitsix(sequence, size = 6):
	return [sequence [x : x+size].lower() for x in range (len(sequence) - size +1)]


db['words'] = db.apply(lambda x: splitsix(x['sequence']), axis=1)
db = db.drop('sequence', axis=1)

print(db.head())


#Since we are going to use scikit-learn natural language processing tools to do the k-mer counting, we need to now convert the lists of k-mers for each gene into string sentences of words that the count vectorizer can use. We can also make a y variable to hold the class labels. Let's do that now.

# now joing the sequence

dog_texts = list(db['words'])
for item in range (len(dog_texts)):
	dog_texts[item] = ' '.join(dog_texts[item])

y_data = db.iloc[:, 0].values


#now we will use bag of words using counter vectorization

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(ngram_range = (4,4))
x = cv.fit_transform(dog_texts)

db['class'].value_counts().sort_index().plot.bar()
#plt.show()

# spliting the dataset


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y_data, test_size = 0.2, random_state = 40)

## using the classifier to classify, iwil use naive bais classifier for reference also random forest after that

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB(alpha = 0.1)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print(y_pred)


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print("Confusion matrix\n")
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))