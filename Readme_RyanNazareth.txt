
######## MSc project titled: Application of Natural Language Processing and Visualisation for Tracking Twitter Discussion Around Global Snakebite Disease #####################
###### Student: Ryan Nazareth, Course: MSc Data Science 2016-2017, Supervisor: Cagatay Turkay #####################

The supplementary zip folder 'Code' contains folders with all the code used in the project and are split into five sections:


1) Tweet_Processing folder

This contains the following files:
              a) twitter_textprocessing.py (tested on Python version 3.6) - running this file will acquire tweets from the past  7 days on topics concerning snakebite.
                The tweets will be stored in a new csv called 'snakebite.csv'

              b) location_geocoding.RMD (R markdown file tested on R studio Version 1.0.44) - This file produces geocoded data from strings in the location.

              c) snakebite04012018.csv - dataset containing all the acquired tweets from 3rd July 2017 to 3rd January 2018 after running both the above files 
                 during the 6 month period 



2) Exploratory_Analysis folder (for the graphs in section 4.1 in the thesis)

This contains the following main files:
             a) exploratory_analysis_snakebite.RMD (R markdown file tested on R studio Version 1.0.44) - Running this produces the graphs in figures 11 and 13 in the thesis 

             b) word2vec_tsne (tested on Python version 3.6) - Running this produces the visualisation for word embeddings in Plotly (figure 14 in the thesis). An example of the output is in                 tsne_plot.html

             c) Tableau_charts (tested on version 10.4) - Contains the stacked bar chart in figure 12 of the thesis.  

             d) snakebite04012018.csv - dataset containing all the acquired tweets from 3rd July 2017 to 3rd January 2018 



3) LDA folder (this contains the data for the results in section 4.2 in the thesis) 

            a) LDA_in_R.Rmd (R markdown file tested on R studio Version 1.0.44) - Topic Modelling in R using LDA() function and LDAvis() to produce interactive LDA visualisation plot in figure              15.

            b) Outputs from running LDA_in_R.Rmd (LDAGibbs5lda_topics.csv, LDAGibbs5topic_probabilities.csv , LDAGibbs10lda_topics.csv , LDAGibbs10topic_probabilities.csv ) 
             These outputs contain results for the topics and topic probabilites for different runs of LDA with 5 and 10 topics. This has been used in table 5 in the thesis.

            c) snakebite04012018.csv - dataset containing all the acquired tweets from 3rd July 2017 to 3rd January 2018 
 

4) Sentiment Analysis folder (this contains the script amd results for section 4.3 in the thesis) 

            a) Sentiment_Analysis.ipynb (Jupyter notebook) - Contains the code and outputs for sentiment analysis for producing the results in Table 6, 7 and figure 16.

            b) Sentiment_Analysis.html- The jupyter notebook above saved in html for viewing the code and output

            c) snakebite04012018.csv - dataset containing all the acquired tweets from 3rd July 2017 to 3rd January 2018 


5) Visualisation_App folder (this contains the script amd results for section 4.4 in the thesis) 

This contains the files for producing the SnakebiteViz visualisation.

          a) index.html - contains links to the various external libraries (D3, Highstocks, JQuery) and javascript files listed below
          b) style.css - for styling the visualisation 
          c) map.js - Javascript file for generating map and favourites bar chart in SnakebiteViz 
          d) tweets_time - Javascript file for generating the interactive tweet frequency and sentiment time series chart 
          e) world-50m.json and world-topo-min.json - json files for generating the world map  

The visualisation can be views in a number of ways:

           Go to my City University Personal Webspace http://www.student.city.ac.uk/~acrz827/ in Google Chrome on a DESKTOP COMPUTER with 100% zoom setting

                   OR


          Open the index.html file listed in MOZILLA FIREFOX browser (downloadable from https://www.mozilla.org/en-GB/firefox/new/)
          on a DESKTOP COMPUTER with a zoom setting at 100%. The visualisation should be viewable directly without the need for setting up a local server.  


                   OR

          Alternatively, for setting up a local server to open in any browser, please follow the steps below:
          a) install nodejs (https://nodejs.org/en/) 
          b) in the terminal run 'npm install -g httpserver'
          c) Change directory to the path where the Visualisation_App folder is downloaded in 
d) run 'httpserver' on the command line to fire up a local server. Go to one of the addresses and you will be able to see the visualisation

The following link (https://www.npmjs.com/package/httpserver) gives a brief summary of the steps above 