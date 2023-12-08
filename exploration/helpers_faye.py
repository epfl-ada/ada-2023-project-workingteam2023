
import pandas as pd
import numpy as np
import datetime
import requests


def valid_format(date_string, date_format='%Y-%m-%d'):
    """
    check if the string in input is in the given format

    params: 
        date_string: string representing a date
        date_format: format of the date string for comparison

    return: 
        True if the date string is in the format YYYY-MM-DD
        False otherwise
    """

    try:
        datetime.datetime.strptime(date_string, date_format)
        return True
    except ValueError:
        return False
    

    

def keep_the_year(date_full, key):
    """
    convert a column containing date with different format keeping only the year

    params: 
        date_full: a dataframe containing a column with date in different format
        key: the name of the column containing the date

    return:
        a dataframe containing only the year of each date
    """
    
    # Converting tyhe date full column to a dataframe, and replacing the nan with null values
    date_full = pd.DataFrame(date_full) 
    date_full  = date_full.fillna(" ")
    
    # creating a dataframe with the same size as date_full, but with nan values
    date_formated = pd.DataFrame(np.nan, index=date_full.index, columns=date_full.columns) 

    # Define date formats
    format1 = '%Y-%m-%d'
    format2 = '%Y-%m'
    format3 = '%Y'

    # Iterate through the values in the datefull column, checking which format it matches, converting it to datetime format, and keeping only the year
    for (index,i) in enumerate(date_full[key]):

        # If the date is out of bounds, consider it as missing value and continue
        if i > '2023' or i < '1800' or i == ' ':    # Even with different date formats the inequality works 
            date_formated[key][index] = np.nan
            continue

        if valid_format(i, format1):
            date_formated[key][index] = datetime.datetime.strptime(i, format1).date().year
        elif valid_format(i, format2):
            date_formated[key][index] = datetime.datetime.strptime(i, format2).date().year
        elif valid_format(i, format3):
            date_formated[key][index] = datetime.datetime.strptime(i, format3).date().year
        else: 
            date_formated[key][index] = np.nan
        
    #Converting to int
    date_formated = date_formated.astype('Int64')

    return date_formated




def link_tconst_freebaseID():
    """
    Wikidata query to get the link between IMDb tconst and freebaseID

    params:
        -

    return: 
        a dataframe containing IMDb tconst and the corresponding freebase ID of the a movie
    """

    # Wikidata SPARQL endpoint
    url = 'https://query.wikidata.org/sparql'

    # Query to get freebase ID and IMDb ID
    # wdt:P345 IMDb ID in wikidata
    # wdt:P646 Freebase ID in wikidata
    query = """
    SELECT ?item ?tconst ?freebaseID WHERE {
        ?item wdt:P345 ?tconst.
        OPTIONAL {?item wdt:P646 ?freebaseID}
    }
    """

    # Query
    params = {'query': query, 'format': 'json'}
    data = requests.get(url ,params = params).json()

    # Create a dataframe that link IMDb tconst and freebaseID
    tconst = []
    freebase_id = []
    for item in data['results']['bindings']:
        tconst.append(item['tconst']['value'])
        freebase_id_val = item.get('freebaseID', {}).get('value', np.nan)
        freebase_id.append(freebase_id_val)

    return pd.DataFrame(data={'tconst': tconst, 'Freebase movie ID': freebase_id})

def print_lda_infos(lda, count_vectorizer, count_data, all_movies):
    """
    Print the topics found by the LDA model
    
    params:
        lda: the LDA model
        count_vectorizer: the count vectorizer
        count_data: the count data
    """
    # Print the topics found by the LDA model
    print("Number of topics:" + str(lda.n_components) + "\n")
    print("Topics found via LDA:")
    for topic_idx, topic in enumerate(lda.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([count_vectorizer.get_feature_names_out()[i]
                        for i in topic.argsort()[:-10 - 1:-1]]))
        
    # for each topic, find the most relevant documents
    n_top_docs = 3
    topic_values = lda.transform(count_data)

    # Create a dataframe with the top n documents in each topic
    top_docs = pd.DataFrame()
    for topic_idx, topic in enumerate(lda.components_):
        top_docs[str(topic_idx)] = topic_values[:,topic_idx].argsort()[:-n_top_docs-1:-1]

    # Add the top documents to the dataframe
    for col in top_docs.columns:
        top_docs[col] = top_docs[col].apply(lambda x: (all_movies.iloc[x]['Summary'], all_movies.iloc[x]['Movie genres']))

    # Print the top documents for each topic
    print("Top documents for each topic:")
    for topic_idx, topic in enumerate(top_docs.columns):
        print("Topic %d:" % (topic_idx))
        print(top_docs[topic].values)
    
    print("\n")
    print("---------------------------------------------------------")
    print("\n")

def get_top_words(lda, count_vectorizer, n_words=3):
    """
    Print the n top words accross all topics
    params:
        lda: the LDA model
        count_vectorizer: the count vectorizer
        n_words: the number of top words to print
    """
      
    # Get the feature names (words) from the vectorizer
    feature_names = count_vectorizer.get_feature_names_out()

    # Initialize a dictionary to store the total word counts across all topics
    total_word_counts = {}

    # Iterate over topics
    for topic_idx, topic in enumerate(lda_.components_):
        # Get the top words for the current topic
        top_word_indices = topic.argsort()[:-n_words-1:-1]
        top_words = [feature_names[i] for i in top_word_indices]

        # Update total word counts
        for word in top_words:
            total_word_counts[word] = total_word_counts.get(word, 0) + 1

    # Sort the words by their total counts
    sorted_words = sorted(total_word_counts.items(), key=lambda x: x[1], reverse=True)

    # Get the top 3 words
    top_words = [word for word, count in sorted_words[:n_words]]

    return top_words