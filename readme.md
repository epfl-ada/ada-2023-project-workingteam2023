# Title 
From Alien Invasions to Global Pandemics: Exploring the Evolution of Film Themes in Response to Societal Fears  

# Abstract 
# A 150 word description of the project idea and goals. What’s the motivation behind your project? What story would you like to tell, and why? 
In this project, we will analyze the chronological evolution of movies’ themes in relation to the society’s fears. Fears have evolved significantly over the decades, shifting from the apprehension of war in the mid-20th century to concerns about emerging technologies during the industrial era, and more recently, to anxieties surrounding pandemics and climate change. By analyzing movies’ emerging themes in the plot, the evolution of movies’ genre as well as the movies with the biggest box office revenue, we want to find if and how the movie industry responded to the emergence and disappearance of major fears in the society. This analysis can be conducted on a global scale as well as on a regional scale, revealing geo-political nuances and historical particularities.  Our analysis will rely on various data sources that depict the societal fears experienced during the 20th and 21st centuries. (139 mots)  

# Research Questions: A list of research questions you would like to address during the project. 
1. What are the primary domains of fear explored in the database's movies, and how do they evolve chronologically ? 
2. Are emerging fears covered in these movies related to historical events ? 
3. Do movies addressing current societal fears tend to achieve higher box office revenue compared to those exploring other themes ? 
4. What is the geographical distribution of the different fears adressed in international movies ? 
5. How has the number of movies addressing major global fears evolved since the dataset's inception ? 
6. What patterns emerge in the portrayal of fears in movies, and are there recurring combinations of fears frequently depicted on screen ?  

# Proposed additional datasets (if any): List the additional dataset(s) you want to use (if any), and some ideas on how you expect to get, manage, process, and enrich it/them. Show us that you’ve read the docs and some examples, and that you have a clear idea on what to expect. Discuss data size and format if relevant. It is your responsibility to check that what you propose is feasible. 
1. Chapman University Survey of American Fears (CSAF) -> pas sûr encore 
This dataset contains an evaluation of American's top fears for years 2014-2023. This data enables us to have a general view on dominant fears in society.   

# Methods 
1. Natural Langage Processing (NLP) : 
2. Unsupervised Topic Modeling : We want to identify sensible topic classes in the movies' plots. This assumes that the number of topics is significantly smaller than the number of movies' plots, which is safe to assume since we have aproximately 45000 movies and we expect to have less than 100 topics. 
- Various approach are possible : 
1. Latent Semantic Analysis (LSA) : This method is based on Singular Value Decomposition (SVD) and is a classical approach to topic modeling. We use a TF-IDF vectorizer to transform the movies' plots into a matrix of TF-IDF features. Then, we apply SVD to this matrix to obtain a matrix of latent topics.
2. Latent Dirichlet Allocation (LDA) : This method is based on a probabilistic approach. We use a CountVectorizer to transform the movies' plots into a matrix of token counts. Then, we apply LDA to this matrix to obtain a matrix of latent topics.

# Proposed timeline 
1. Extract the general domains of fear in the global society from additional datasets, thus allowing us to have a set of fears to look for in the movies’ plots.
2. Perform NLP and apply topic modeling techniques on the plot to extract movies that treat those fears. Class them by fear domains. 
3. Analyze the data from those movies (number of movies, box office revenue, release date) 
4. Draw plots depicting our results 
5. Create the website  

# Organization within the team: A list of internal milestones up until project Milestone P3. 
Octavio and Faye were responsible for writing the README file.  
Clara, Romain and Colin were responsible for writing the Notebook. Clara focused on the general handling pipelines. Romain did some research on potential additional dataset and contributed to the data handling pipelines part. Colin did some research on NLP methods and contributed to the data handling pipelines part. 
We all did some personal research on the subject.  

# Questions for TAs (optional): Add here any questions you have for us related to the proposed project.
