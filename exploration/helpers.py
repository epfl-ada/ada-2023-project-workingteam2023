
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
    convert a column containing date with different format keeping only the year using the pandas apply function
    
    params: 
        date_full: a dataframe containing a column with date in different format
        key: the name of the column containing the date
        
    return:
        a dataframe containing only the year of each date, in place of the original column
    """

    def keep_the_year_apply_helper(date_string):
        """
        helper function for keep_the_year_apply
        """
        # Define date formats
        format1 = '%Y-%m-%d'
        format2 = '%Y-%m'
        format3 = '%Y'

        # If the date is out of bounds, consider it as missing value and continue
        if str(date_string) > '2023' or str(date_string) < '1800' or str(date_string) == ' ' or str(date_string) == 'nan':    # Even with different date formats the inequality works
            return np.nan

        if valid_format(date_string, format1):
            return datetime.datetime.strptime(date_string, format1).date().year
        elif valid_format(date_string, format2):
            return datetime.datetime.strptime(date_string, format2).date().year
        elif valid_format(date_string, format3):
            return datetime.datetime.strptime(date_string, format3).date().year
        else: 
            return np.nan
        
    # Apply the helper function to the column
    date_full[key] = date_full[key].apply(keep_the_year_apply_helper).astype('Int64')

    return date_full




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