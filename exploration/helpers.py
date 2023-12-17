
import pandas as pd
import numpy as np
import datetime
import requests
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('averaged_perceptron_tagger')




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



def get_cleaned_data(path):
    """
    Get the data from the given path and clean it

    params:
        path: the path of the folder containing the data

    return:
        the cleaned datasets
    """
    print("Loading the data...")
    #load data/moviesummaries/character.metadata.tsv
    character_metadata = pd.read_csv(path + 'moviesummaries/character.metadata.tsv', sep='\t', header=None)
    character_metadata.columns = ["Wikipedia movie ID", "Freebase movie ID", "Movie release date", "Character name", "Actor date of birth", "Actor gender", 
                                "Actor height", "Actor ethnicity", "Actor name", "Actor age", "Freebase character/actor map ID", 
                                "Freebase character ID", "Freebase actor ID"]

    #load data/moviesummaries/plot_summaries.txt
    plot_summaries = get_summaries(path)

    #load data/moviesummaries/movie.metadata.tsv
    movie_metadata = pd.read_csv(path + 'moviesummaries/movie.metadata.tsv', sep='\t', header=None)
    movie_metadata.columns = ["Wikipedia movie ID", "Freebase movie ID", "Movie name", "Movie release date", "Movie revenue", "Movie runtime",
                            "Movie languages", "Movie countries", "Movie genres"]

    #load data/moviesummaries/name.clusters.txt
    name_clusters = pd.read_csv(path + 'moviesummaries/name.clusters.txt', sep='\t', header=None)
    name_clusters.columns = ["Character name", "Freebase character/actor map ID"]


    print("Cleaning the data...")
    # Merge 'left' the movie_metadata and plot_summaries dataframes on the Wikipedia movie ID column
    all_movies = movie_metadata.merge(plot_summaries, on="Wikipedia movie ID", how="left")


    # drop one of each pair of duplicates
    all_movies.drop_duplicates(subset=["Movie name", "Movie release date", "Movie revenue", "Movie languages", "Movie genres", "Movie countries", "Movie runtime", "Summary"], inplace=True, keep="first")
    

    #Converting the movie release date to keep only the year for the all_movie table
    all_movies = keep_the_year(all_movies, key='Movie release date')


    # Some columns contains dicts. Let's only keep the values of these dicts as lists since we don't care about their keys
    all_movies['Movie genres'] = [list(eval(genre).values()) for genre in all_movies['Movie genres']]
    all_movies['Movie languages'] = [list(eval(genre).values()) for genre in all_movies['Movie languages']]
    all_movies['Movie countries'] = [list(eval(genre).values()) for genre in all_movies['Movie countries']]


    #Converting the dates to keep only the year for the character_metadata table
    character_metadata = keep_the_year(character_metadata, key='Movie release date')
    character_metadata = keep_the_year(character_metadata, key='Actor date of birth')


    print("Adding IMDb ratings...")
    # Add IMDb ratings
    movie_ratings = pd.read_csv(path + 'title.ratings.tsv', sep='\t', header=0)

    # Create the table
    link_id = link_tconst_freebaseID()

    # Drop duplicates
    link_id = link_id.drop_duplicates(subset=['tconst'])
    link_id = link_id.drop_duplicates(subset=['Freebase movie ID'])

    # Add freebase ID to movie_ratings
    movie_ratings = pd.merge(movie_ratings, link_id, on='tconst', how='left')

    # Merge all_movies and movie_ratings
    all_movies = pd.merge(all_movies, movie_ratings, on='Freebase movie ID', how='left')
    # Drop tconst column
    all_movies.drop(columns=['tconst'], inplace=True)

    return all_movies, character_metadata, name_clusters



def get_summaries(path, punctuation=True, stop_words=True, lemmatize=True, pos_tag=True, movie_film=True, remove_names = True, force_reload=False, save=True):
    print("Loading and cleaning the summaries...")

    array_names = [
    "john", "robert", "james", "william", "mary", "patricia", "jennifer", "linda", "elizabeth", "barbara",
    "michael", "david", "richard", "joseph", "thomas", "charles", "christopher", "daniel", "matthew", "anthony",
    "donald", "mark", "paul", "steven", "andrew", "kenneth", "george", "joshua", "brian", "kevin",
    "ashley", "jessica", "amanda", "sarah", "melissa", "emily", "nicole", "kayla", "stephanie", "michelle",
    "jacob", "christopher", "matthew", "joshua", "nicholas", "andrew", "daniel", "tyler", "joseph", "david",
    "sophia", "emma", "olivia", "ava", "isabella", "mia", "abigail", "emily", "madison", "elizabeth",
    "hannah", "amelia", "ella", "grace", "chloe", "charlotte", "avery", "sofia", "harper", "mila",
    "ethan", "michael", "alexander", "william", "james", "christopher", "daniel", "matthew", "andrew", "joseph",
    "david", "ryan", "jacob", "nicholas", "tyler", "emily", "madison", "emma", "hannah", "olivia", "alice", "tom", "jerry",
    "frank", "joe",
    "aarav", "aryan", "vivaan", "vihaan", "advait", "arjun", "reyansh", "ishaan", "kabir", "ritvik",
    "ananya", "aaradhya", "pari", "kiara", "siya", "dia", "aisha", "riya", "avni", "zara",
    "mohammed", "aman", "armaan", "aryan", "ishaan", "vihaan", "advait", "reyansh", "kabir", "vivaan",
    "sanskriti", "ananya", "aaradhya", "pari", "kiara", "siya", "dia", "aisha", "riya", "avni",
    "raj", "vijay", "kumar", "ramesh", "amit", "rakesh", "suresh", "pradeep", "vinod", "sunil",
    "neeta", "sangeeta", "renu", "anita", "preeti", "neha", "seema", "meena", "suman", "arti",
    "rahul", "amit", "vikas", "rajesh", "deepak", "sanjay", "ranjeet", "santosh", "ravi", "anil",
    "smita", "pooja", "neha", "priya", "arti", "ritu", "swati", "anita", "kavita", "sangeeta",
    "varun", "aditya", "arjun", "rohan", "rajat", "vishal", "vivek", "alok", "manish", "amit"
    ]

    #check if processed_summaries.tsv exists
    try:
        if force_reload:
            raise FileNotFoundError
        plot_summaries = pd.read_csv(path + 'moviesummaries/processed_summaries.tsv', sep='\t', header=0)
        print("Summaries loaded from processed_summaries.tsv")
        return plot_summaries
    except FileNotFoundError:
        print("processed_summaries.tsv not found, processing the summaries...")

    #load data/moviesummaries/plot_summaries.txt
    plot_summaries = pd.read_csv(path + 'moviesummaries/plot_summaries.txt', sep='\t', header=None)
    plot_summaries.columns = ["Wikipedia movie ID", "Summary"]

    # summaries
    # First we copy all_movies and filter to only keep movies with a plot summary
    plot_summaries = plot_summaries.dropna(subset=['Summary'])

    # tokenize
    print("Tokenizing...")
    plot_summaries['Summary'] = plot_summaries['Summary'].apply(lambda x: word_tokenize(x))

    # remove punctuation
    if punctuation:
        print("Removing punctuation...")
        plot_summaries['Summary'] = plot_summaries['Summary'].apply(lambda x: [word for word in x if word not in string.punctuation])

    '''
    ORIGINAL
    # remove stop words
    if stop_words or movie_film:
        print("Removing stop words / movie_film...")
        stop_words = set()
        if stop_words:
            stop_words = set(stopwords.words('english'))
        if movie_film:
            stop_words.add('film')
            stop_words.add('movie')
        stop_words = set(stopwords.words('english'))
        plot_summaries['Summary'] = plot_summaries['Summary'].apply(lambda x: [word for word in x if word.lower() not in stop_words])
    '''

    # MODIFICATION
    # remove stop words and common words
    if stop_words:
        print("Removing stop words and common words/names...")
        stop_words = set(stopwords.words('english'))
        if movie_film:
            print("Removing common words...")
            stop_words.update(['film', 'films', 'movie', 'movies'])  # Use update to add multiple elements           
        if remove_names:
            print("Removing common names...")
            stop_words.update(array_names)
        plot_summaries['Summary'] = plot_summaries['Summary'].apply(lambda x: [word for word in x if word.lower() not in stop_words])

    ''' 
    # POS tag to remove NNP (proper nouns) and NNPS (plural proper nouns)
    if pos_tag:
        print("POS tagging...")
        plot_summaries['Summary'] = plot_summaries['Summary'].apply(lambda x: nltk.pos_tag(x))
        plot_summaries['Summary'] = plot_summaries['Summary'].apply(lambda x: [word for word, tag in x if tag != 'NNP' and tag != 'NNPS'])
    '''

    # lemmatize
    if lemmatize:
        print("Lemmatizing...")
        lemmatizer = nltk.stem.WordNetLemmatizer()
        plot_summaries['Summary'] = plot_summaries['Summary'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

    # join
    plot_summaries['Summary'] = plot_summaries['Summary'].apply(lambda x: ' '.join(x))

    # save
    if save:
        print("Saving the processed summaries...")
        plot_summaries.to_csv(path + 'moviesummaries/processed_summaries.tsv', sep='\t', index=False)

    return plot_summaries
