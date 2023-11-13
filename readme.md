# Title 
From Alien Invasions to Global Pandemics: Exploring the Evolution of Film Themes in Response to Societal Fears  

# Abstract 
# A 150 word description of the project idea and goals. What’s the motivation behind your project? What story would you like to tell, and why? 
In this project, we will analyze the chronological evolution of movies’ themes in relation to the society’s fears. Fears have evolved significantly over the decades, shifting from the apprehension of war in the mid-20th century to concerns about emerging technologies during the industrial era, and more recently, to anxieties surrounding pandemics and climate change.   
By analyzing movies’ emerging themes in the plot, the evolution of movies’ genre as well as the movies’ success (based on the IMDb rating), we can depict how the movie industry responded to the emergence and disappearance of major fears in the society. This analysis can be conducted on a global scale as well as on a regional scale.  
Examining the patterns of societal fears provides a deeper insight into the broader aspects of society. It reflects the historical, political and cultural context of the world across the years. (143 mots)

# Research Questions: A list of research questions you would like to address during the project. 
1.	What are the primary domains of fear explored in the database's movies, and how do they evolve chronologically? 
2.	Do movies addressing current societal fears tend to achieve higher box office revenue compared to those exploring other themes? 
3.	How has the number of movies addressing major global fears evolved since the dataset's inception? 
4.	Are emerging fears covered in movies related to historical, political or cultural events?
5.	What is the geographical distribution of the different fears addressed in international movies? 
6.	What patterns emerge in the portrayal of fears in movies, and are there recurring combinations of fears frequently depicted on screen?
7.	What evolution of the society can we depict from all of the previous results?

# Proposed additional datasets (if any): List the additional dataset(s) you want to use (if any), and some ideas on how you expect to get, manage, process, and enrich it/them. Show us that you’ve read the docs and some examples, and that you have a clear idea on what to expect. Discuss data size and format if relevant. It is your responsibility to check that what you propose is feasible. 
1. IMDb: Incorporating the IMDb dataset into our analysis provides additional insights into movies. We will use the weighted average ratings data from title.ratings.tsv.gz to quantify a movie's success. To achieve this, we first plan to merge the data from title.ratings.tsv.gz with the data from title.basics.tsv.gz, establishing a connection between alphanumeric identifiers and movie titles. Subsequently, we will merge this combined data with the CMU movie summary corpus based on movie titles. However, managing duplicate movie titles poses a challenge that still needs to be addressed in the process.

# Methods

# Proposed timeline 
1. Extract the general domains of fear in the global society from additional datasets 
2. Perform NLP on the plot to extract movies that treat those fears. Class them by fear domains. 3. Analyze the data from those movies (number of movies, box office revenue, release date) 
4. Draw plots depicting our results 
5. Create the website  

# Organization within the team: A list of internal milestones up until project Milestone P3. 
Octavio and Faye were responsible for writing the README file.  
Clara, Romain and Colin were responsible for writing the Notebook. Clara focused on the general handling pipelines. Romain did some research on potential additional dataset and contributed to the data handling pipelines part. Colin did some research on NLP methods and contributed to the data handling pipelines part. 
We all did some personal research on the subject.  

# Questions for TAs (optional): Add here any questions you have for us related to the proposed project.
