# Title 
From Alien Invasions to Global Pandemics: Exploring the Evolution of Film Themes in Response to Societal Fears  

# Abstract 
# A 150 word description of the project idea and goals. What’s the motivation behind your project? What story would you like to tell, and why? 
In this project, we will analyze the chronological evolution of movies’ themes in relation to society’s fears. Fears have evolved significantly over the decades, shifting from the apprehension of war in the mid-20th century to concerns about emerging technologies during the industrial era, and more recently, to anxieties surrounding pandemics and climate change. By analyzing movies’ emerging themes in the plot, the evolution of movies’ genre as well as the movies with the highest popularity, we can depict how the movie industry responded to the emergence and disappearance of major fears in the society. This analysis can be conducted on a global scale as well as on a regional scale, revealing geo-political nuances and historical particularities.  Our analysis will rely on various data sources that depict the societal fears experienced during the 20th and 21st centuries.

# Research Questions: A list of research questions you would like to address during the project. 
1. What are the primary domains of fear explored in the database's movies, and how do they evolve chronologically? 
2. Do movies addressing current societal fears tend to have higher IMDb ratings compared to those exploring other themes? 
3. How has the number of movies addressing major global fears evolved since the dataset's inception? 
4. Are emerging fears covered in movies related to historical, political or cultural events?
5. What is the geographical distribution of the different fears addressed in international movies? 
6. What patterns emerge in the portrayal of fears in movies, and are there recurring combinations of fears frequently depicted on screen?
7. What evolution of the society can we depict from all of the previous results?


# Proposed additional datasets (if any): List the additional dataset(s) you want to use (if any), and some ideas on how you expect to get, manage, process, and enrich it/them. Show us that you’ve read the docs and some examples, and that you have a clear idea on what to expect. Discuss data size and format if relevant. It is your responsibility to check that what you propose is feasible. 
1. IMDb: Incorporating the IMDb dataset into our analysis provides additional insights into the movies. Since there is a lot of missing values for the movies' box office revenues, will use the weighted average ratings data from title.ratings.tsv.gz to quantify a movie's success. IMDb is a recognised source for movie reviews. We therefore trust their methods for collecting and weighting this data. 
The IMDb dataset and our dataset use different identifiers for movies, IMDb employs "tconst," while our dataset uses the Freebase movie ID. In order to merge these datasets, we must establish a link between the two sets of IDs. We retrieve the correspondance from the Wikidata query service by performing a query in SPARSQL. Subsequently, we generate a correspondence table, removing any duplicate entries. With this completed correspondence table, we can proceed to merge the two datasets seamlessly.


# Methods 
Appart from the classic preprocessing methods that we used to filter and arrange our data, we used Natural Language Processing (NLP), and in particular the Latent Dirichlet Allocation (LDA), which is a generative statistical model used to classify text in a document to a particular topic. It builds a topic per document model and words per topic model, modeled as Dirichlet distributions. We used the LDA to look for particular fears in the plot summaries of the movies, namely the fear of war, of climate change, of corruption, of terrorism, of civilization collapse, of pandemic, of technology and of aliens.

# Proposed timeline 
1. Extract the general domains of fear in the global society from articles
2. Process the original data: clean, merge and display general interesting features of our data
3. Process the additional data: clean and merge it with the original data
4. Perform NLP on the plot to extract movies that treat those fears. Class them by fear domains. 
5. Analyze the data from those movies (number of movies, release date, IMDb reviews, ...) 
4. Draw plots depicting interesting trends from our results
5. Create the website, display all our interesting results and draw a conclusion of our analysis
 

# Organization within the team: A list of internal milestones up until project Milestone P3. 
Octavio and Faye were responsible for writing the README file.  
Clara, Romain and Colin were responsible for writing the Notebook. Clara focused on cleaning, merging and drawing general trends of the orginal dataset. Romain was responsible of handling the additional dataset. Colin started the NLP analysis on the plot summaries.
We all did some personal research on the subject and helped eachother for our respective parts.  


# Questions for TAs (optional): Add here any questions you have for us related to the proposed project.
