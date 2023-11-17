
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