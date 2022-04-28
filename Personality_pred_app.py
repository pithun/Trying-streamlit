import pandas as pd
import joblib
import streamlit as st

# Data Cleaning and preprocessing 
import numpy as np
import nltk
import re
from wordcloud import STOPWORDS
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.preprocessing import LabelEncoder

durl= ('https://raw.githubusercontent.com/pithun/Trying-streamlit/main/mbti_1.csv') 
personality = pd.read_csv('mbti_1.csv')

labenc = LabelEncoder()
personality['type_encoded'] = labenc.fit_transform(personality.type)
# App title
st.title("PERSONALITY PREDICTION using Myers-Briggs Type Indicator (MBTI)")

st.text('\n')

st.subheader('The following test is designed to measure your MBTI Personality type')

st.text('Several hints about how to best complete this survey:' +'\n'+
    '''    • There are no right answers to any of these questions.
    • Do not over-analyze the questions. Some seem worded poorly.
      Go with what feels best.
    • Answer the questions as “the way you are”, not “the way you’d like to be seen 
      by others”''')

st.markdown(f'<h1 style="color:#FF6347;font-size:24px;">{"Wanna know more about how we are making our Prediction ?"}</h1>', unsafe_allow_html=True)

with st.expander('Expand this by clicking to know more'):
    st.text('''The data was collected from the posts of people in the personality cafe which 
is a community where you can.
    • Follow topics that matter to you
    • Connect with those who share your interests
    • Learn from the experts in the community you can also ask, comment, and 
    connect! \n
After collection of data using techniques in a branch of Machine learning called
Natural languaege processing we were able to train a model on that data and make
these predictions. Pretty cool right ?!!''')
    
st.markdown(    """
<style>
span[data-baseweb="tag"] {
  background-color: blue !important;
}
</style>
""",  unsafe_allow_html=True)


questions = ['1. At a party do you: ', '2. Are you more: ', '3. Is it worse to: ', '4. Are you more impressed by: ', '5. Are more drawn toward the: ', '6. Do you prefer to work:', '7. Do you tend to choose: ', '8. At parties do you: ', '9. Are you more attracted to: ', '10. Are you more interested in: ', '11. In judging others are you more swayed by: ', '12. In approaching others is your inclination to be somewhat: ', '13. Are you more: ', '14. Does it bother you more having things: ', '15. In your social groups do you: ', '16. In doing ordinary things are you more likely to: ', '17. Writers should: ', '18. Which appeals to you more: ', '19. Are you more comfortable in making: ', '20. Do you want things: ', '21. Would you say you are more: ', '22. In phoning do you: ', '23. Facts: ', '24. Are visionaries: ', '25. Are you more often: ', '26. Which is more admirable: ', '27. Do you feel better about: ', '28. In company do you: ', '29. Common sense is: ', '30. Children often do not: ', '31. In making decisions do you feel more comfortable with:', '32. Are you more:']

opt1 = ['Interact with many, including strangers ', 'Realistic than speculative', 'Have your “head in the clouds”', 'Principles', 'Convincing', 'To deadlines', 'Rather carefully', 'Stay late, with increasing energy', 'Sensible people', 'What is actual', 'Laws than circumstances', 'Objective', 'Punctual', 'Incomplete', 'Keep abreast of other’s happenings', 'Do it the usual way', '“Say what they mean and mean what they say”', 'Consistency of thought', 'Logical judgment', 'Settled and decided', 'Serious and determined', 'Rarely question that it will all be said', '“Speak for themselves”', 'somewhat annoying', 'a cool-headed person', 'the ability to organize and be methodical', 'having purchased', 'initiate conversation', 'rarely questionable', 'make themselves useful enough', 'standards', 'firm than gentle']

opt2 = ['Interact with a few, known to you', 'Speculative than realistic', 'Be “in a rut”', 'Emotions', 'Touching', 'Just “whenever”', 'Somewhat impulsively', 'Leave early with decreased energy', 'Imaginative people', 'What is possible', 'Circumstances than laws', 'Personal', 'Leisurely', 'Completed', 'Get behind on the news', 'Do it your own way', 'Express things more by use of analogy', 'Harmonious human relationships', 'Value judgments', 'Unsettled and undecided', 'Easy-going', 'Rehearse what you’ll say', 'Illustrate principles', 'rather fascinating', 'a warm-hearted person', 'the ability to adapt and make do', 'having the option to buy', 'wait to be approached', 'frequently questionable', 'exercise their fantasy enough', 'feelings', 'gentle than firm']

st.text('\n'+'Below are {} questions to be answered by you. After answering, hit the \n"What\'s my Personality" button and wait a while to know WHAT TRAITS YOU POSSESS'.format(len(opt2)))

new = ''
for a,b,c in zip(questions, opt1, opt2):
    cont = st.radio(a, (b, c))
    new+=cont+' '
    
#st.write(new)


stopwords_better = list(STOPWORDS)
for remove in ['\'', '^', '^^', 'doe', 'ha', 'wa', '\'s', '\'the', '\'i']:
    stopwords_better.append(remove)

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

for a in stopwords.words('english'):
    stopwords_better.append(a)
stopwords_better = list(set(stopwords_better))

# The code below basically adds escape characters  \\ to the emoticon characters since regex is used to
# remove them.
# for example, we have :-) but ) is a group character in regex so, we had to escape it 
new_string = ''
new_emoticons = []
meanings = []
to_escape = [']', '[', '+', '*', '(', ')', '?', '{', '}', '|', '.'] 
for a, mean in EMOTICONS_EMO.items():
    meanings.append(mean)
    #print(a)
    for each in a:
        if each in to_escape:
            new_string +=  '\\'+each
        else:
            new_string+=each
    new_emoticons.append(new_string)
    new_string = ''

# the problem here was that (\\\\) doesn't escape the second parenthesis, so i had to change it to \\)
for ind, new in enumerate(new_emoticons):
    new_emoticons[ind] = new.replace('(\\\\)', '(\\)')
    
    
def emoji_to_text(text):
    '''This function replaces emojis with their meanings'''
    for emoji, meaning in UNICODE_EMOJI.items():
        text = text.replace(emoji, meaning)
    return text

def convert_emoticons(text):
    ''' This function replaces emoticons which are basically symbols to represent expressions with their meaning 
    some are in the posts so it makes sense to remove them.'''
    for emoticon, meaning in zip(new_emoticons, meanings):
        text = re.sub(emoticon, meaning, text)
    return text

def remlinks_symbs(df):
    '''This function creates a new column with the links, mentions, symbols, numbers of more than 1character 
    removed and emojis, emoticons and replaced with their meaning in the data set'''
    df['post'] = df.posts
    
    # replacing link with space 
    # we noticed some links don't have :// it's just https:youtu... and some start from www.
    # example www.youtube.com/watch?v=2o5FHtUaKNQ so we had to tweak the regex multiple times to meet the
    # needs.
    df.post.replace(r'(https?:/?/?|ftp:/?/?|www|[yY]outube)+?[\w\-\?\=\%.]+?.[\w\!\£\$\%\^\&\*\(\)\_\+\-\=\{\\}\~\[\]\#\:\@\;\'\<\>\?\,\.\/\\]+', ' ', regex= True, inplace= True)
    
    # replacing mentions with space
    df.post.replace(r'@[\w\!\£\$\%\^\&\*\(\)\_\+\-\=\{\\}\~\[\]\#\:\@\;\'\<\>\?\,\.\/\\]+', ' ', 
                    regex= True, inplace= True)
    
    #reducing all to lowercase
    df['post'] = df['post'].str.lower()
    
    # Replacing emoticons with their meaning
    df.post = df.post.apply(convert_emoticons)
    
    #replacing emojis with their meanings
    df.post = df.post.apply(emoji_to_text)
    
    # replacing one or more symbols except ', | and _ with space
    # we left the ||| to be able to implement ngrams sentence wise since it indicates boundary of one
    # sentence.
    df.post.replace(r'[^\w^\'^|]+', ' ', regex = True, inplace = True)
    
    #removes numbers longer than two since 4w3 has a meaning removing numbers will leave just w 
    df.post.replace(r'\d\d+', '', regex = True, inplace = True)
    
    #replacing underscores with nothing
    df.post.replace(r'_', '', regex = True, inplace = True)
    return df

# removing apostrophe from beginning and end of posts and leaving them in some words so as to remove 
# stopwords.
def rem_pos(post):
    if post[0] == '\'' and post[-1]=='\'':
        post = post[1:-1]
    elif post[0] == '\'':
        post = post[1:]
    elif post[-1]=='\'':
        post = post[:-1]
    else:
        pass
    return post

def lemmatizee(the_list):
    # Lemmatizer function using the returning default part of speech(noun) for each word passed through it
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(a) for a in the_list.split()]

def lemmatized(sentence):
     # Lemmatizer function using parts of speech tagging.
    lemmatizer = WordNetLemmatizer()
    wnl = WordNetLemmatizer()
    lemmatized = []
    for word, tag in pos_tag(sentence.split()):
        if tag.startswith("NN"):
            lemmatized.append(wnl.lemmatize(word, pos='n'))
        elif tag.startswith('VB'):
            lemmatized.append(wnl.lemmatize(word, pos='v'))
        elif tag.startswith('JJ'):
            lemmatized.append(wnl.lemmatize(word, pos='a'))
        else:
            lemmatized.append(word)
    return lemmatized

def BOW(train_data, test_data, ngrams= 1, max_feat= 3500, split_data = True): 
    '''This does all cleaning and returns the bag of documents vector for both train and test data
    note that you have to manually supply train and test data'''
    
    # for train_data
    # here we apply the clean function above
    train_data = remlinks_symbs(train_data)
    
    #for test_data
    test_data = remlinks_symbs(test_data)
    
    if ngrams == 1:
        train_data.post.replace(r'[^\w^\']+', ' ', regex = True, inplace = True)
        train_data.post = train_data.post.apply(rem_pos)
            
        test_data.post.replace(r'[^\w^\']+', ' ', regex = True, inplace = True)
        test_data.post = test_data.post.apply(rem_pos)
        
        # tokenizing words and making the vectors
        vectorizer=TfidfVectorizer(max_features= max_feat, stop_words= stopwords_better, tokenizer= 
                                   lemmatized, token_pattern= '(?u)\\b\\w+\'\w+\\b')
        # the reason it tokenizes "'cause" is because ' is recognized as a boundary
        
        char_array_tr = vectorizer.fit_transform(train_data.post).toarray()
        char_array_te = vectorizer.transform(test_data.post).toarray()
        
        # Turning into dataframes
        frequency_matrix_tr = pd.DataFrame(char_array_tr, columns= vectorizer.get_feature_names_out())
        frequency_matrix_te = pd.DataFrame(char_array_te, columns= vectorizer.get_feature_names_out())
    else:
        pass
    if split_data== True:
        return (frequency_matrix_tr, frequency_matrix_te)
    else:
        return frequency_matrix_tr 

    
test_data = pd.DataFrame({'posts':[new]})

if st.button('\n What\'s My Personality'):
    with st.spinner('Please wait...'):
        to_predict = BOW(personality, test_data)
        testing_data = to_predict[1]
        #st.write(testing_data)
        
        predictor = joblib.load('Logregmodel.pkl')
        prediction = predictor.predict(testing_data)
        st.write('Standard Scaler predicts.....'+labenc.inverse_transform(prediction))
