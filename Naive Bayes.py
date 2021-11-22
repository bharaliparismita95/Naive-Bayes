import numpy as np
from os import listdir
from os.path import join

# data path
train_data_path = '20_newsgroups/Train_data'
test_data_path = '20_newsgroups/Test_data'

# creating a list of folder names
train_folders = [f for f in listdir(train_data_path)]
test_folders = [f for f in listdir(test_data_path)]

# to store list of all files
train_files = []
for folder_name in train_folders:
    folder_path = join(train_data_path, folder_name)
    train_files.append([f for f in listdir(folder_path)])

test_files = []
for folder_name in test_folders:
    folder_path = join(test_data_path, folder_name)
    test_files.append([f for f in listdir(folder_path)])


# list of path names for all the documents
x_train = []
for fo in range(len(train_folders)):
    for fi in train_files[fo]:
        x_train.append(join(train_data_path, join(train_folders[fo], fi)))
print('Number of train documents:', len(x_train))

x_test = []
for fo in range(len(test_folders)):
    for fi in test_files[fo]:
        x_test.append(join(test_data_path, join(test_folders[fo], fi)))
print('Number of test documents:', len(x_test))

# classes each document belongs to
y_train = []
for folder_name in train_folders:
    folder_path = join(train_data_path, folder_name)
    num_of_files = len(listdir(folder_path))
    for i in range(num_of_files):
        y_train.append(folder_name)

y_test = []
for folder_name in test_folders:
    folder_path = join(test_data_path, folder_name)
    num_of_files = len(listdir(folder_path))
    for i in range(num_of_files):
        y_test.append(folder_name)

# not so important words and punctuations
stopwords = ['is', 'are', 'the', 'was', 'of', 'a', 'and', 'not', 'it', 'on', 'if', 'off', 'most', 'very', 'for', 'by',
             'but', 'under', 'have', 'again', 'all', 'any', 'because', 'been', 'before', 'below', 'both', 'few', 'how',
             'about', 'after', 'am', 'doing', 'your', 'once', 'into', 'he', 'she', 'we', 'his', 'same', 'what', 'when',
             'where', 'whom', 'their', 'theirs', 'should', 'must', 'other', 'more', 'me', 'you', 'may', 'could',
             'would', 'will', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'why', 'every',
             'either', 'let', 'had', 'has', 'only', 'at', 'an', 'were', 'to', 'with', 'as', 'also', 'as', 'from',
             'then', 'than', 'them', 'in', 'above', 'this', 'that', 'which', 'or', 'over', 'while', 'they', 'so',
             'some', 'such', 'do', 'did', 'does', 'each', 'can', 'cannot', 'one', 'our', 'out', 'two', 'three', 'four',
             'five', 'six', 'seven', 'eight', 'nine', 'ten', 'hundred', 'thousand', '1st', '2nd', '3rd', '4th', '5th',
             '6th', '7th', '8th', '9th', '10th', ',', '.', ';', ':', '?', '{', '}', '(', ')', '-', '>', '<', '*', '@',
             '!', '/', '+', '=', '%', '$', '""', "''", '&', '#', '|', '--', '---', '|>', '>>', '<<']


# function to remove the stopwords
def remove_stopwords(words):
    words = [word for word in words if not word in stopwords]
    return words


# function to remove more unwanted words
def remove_words(words):
    # to remove numeric strings
    words = [word for word in words if not word.isdigit()]
    # to remove single character strings
    words = [word for word in words if not len(word) == 1]
    # to remove blanks
    words = [str for str in words if str]
    # to normalize the cases of words
    words = [word.lower() for word in words]
    # to remove words with 2 characters
    words = [word for word in words if len(word) > 2]
    return words


# function to tokenize sentences
def tokenize_sentence(line):
    words = line[0:len(line) - 1].strip().split(" ")
    words = remove_words(words)
    words = remove_stopwords(words)
    return words


# function to tokenize each document
def tokenize_doc(path):
    f = open(path, 'r')
    text_lines = f.readlines()
    doc_words = []
    for line in text_lines:
        doc_words.append(tokenize_sentence(line))
    return doc_words


# function to convert 2D array to 1D
def flatten(l):
    new_list = []
    for i in l:
        for j in i:
            new_list.append(j)
    return new_list


# tokenizing training data
list_of_words = []
for document in x_train:
    list_of_words.append(flatten(tokenize_doc(document)))

# converting the tokens into numpy array
word_array = np.asarray(flatten(list_of_words))

# finding the number of unique words that we have extracted from the documents
words, counts = np.unique(word_array, return_counts=True)

# sorting the unique words according to their frequency
freq, frequent_words = (list(i) for i in zip(*(sorted(zip(counts, words), reverse=True))))
print('No of frequent words:', len(frequent_words))

# taking no of words for features
n = 60000
features = frequent_words[0:n]

# creating a dictionary
dictionary = {}
doc_no = 1
for doc_words in list_of_words:
    doc_words_array = np.asarray(doc_words)
    w, c = np.unique(doc_words_array, return_counts=True)
    dictionary[doc_no] = {}
    for i in range(len(w)):
        dictionary[doc_no][w[i]] = c[i]
    doc_no = doc_no + 1

# train data
X_train = []
for k in dictionary.keys():
    row = []
    for f in features:
        if f in dictionary[k].keys():
            row.append(dictionary[k][f])
        else:
            row.append(0)
    X_train.append(row)

# to convert the data and label into numpy arrays
X_train = np.asarray(X_train)
Y_train = np.asarray(y_train)

# tokenizing test data
list_of_words_test = []
for document in x_test:
    list_of_words_test.append(flatten(tokenize_doc(document)))

# dictionary for test data
dictionary_test = {}
doc_no = 1
for doc_words in list_of_words_test:
    doc_words_array = np.asarray(doc_words)
    w, c = np.unique(doc_words_array, return_counts=True)
    dictionary_test[doc_no] = {}
    for i in range(len(w)):
        dictionary_test[doc_no][w[i]] = c[i]
    doc_no = doc_no + 1

# test data
X_test = []
for key in dictionary_test.keys():
    X_test.append(list(dictionary_test[key].keys()))


# converting test data into numpy array
Y_test = np.asarray(y_test)

# function to create a training dictionary
def fit(X_train, Y_train):
    result = {}
    labels, counts = np.unique(Y_train, return_counts=True)
    for i in range(len(labels)):
        current_class = labels[i]
        result["All_data"] = len(Y_train)
        result[current_class] = {}
        X_train_current = X_train[Y_train == current_class]
        no_features = n
        for j in range(no_features):
            result[current_class][features[j]] = X_train_current[:, j].sum()
        result[current_class]["All_counts"] = counts[i]
    return result


# function to calculate naive bayes probability
def nb_prob(train_dictionary, x, current_class):
    output = np.log(train_dictionary[current_class]["All_counts"]) - np.log(train_dictionary["All_data"])
    no_words = len(x)
    for j in range(no_words):
        if x[j] in train_dictionary[current_class].keys():
            xj = x[j]
            count_current_class__xj = train_dictionary[current_class][xj] + 1
            count_current_class = train_dictionary[current_class]["All_counts"] + len(train_dictionary[current_class].keys())
            xj_prob = np.log(count_current_class__xj) - np.log(count_current_class)
            output = output + xj_prob
        else:
            continue
    return output


# function to calculate class of each test document
def predict_doc(train_dictionary, x):
    classes = train_dictionary.keys()
    prob = -20000
    doc_class = -1
    for current_class in classes:
        if current_class == "All_data":
            continue
        doc_current_class = nb_prob(train_dictionary, x, current_class)
        if doc_current_class > prob:
            prob = doc_current_class
            doc_class = current_class
    return doc_class

# predict function
def predict(train_dictionary, X_test):
    Y_predicted = []
    for x in X_test:
        y_predicted = predict_doc(train_dictionary, x)
        Y_predicted.append(y_predicted)
    return Y_predicted

# training the train data
train_dictionary = fit(X_train, Y_train)

# making predictions on test data
predictions = predict(train_dictionary, X_test)
predictions = np.asarray(predictions)

# calculating the test accuracy
test_accuracy = np.mean(predictions == Y_test)
print('Test accuracy:', test_accuracy)