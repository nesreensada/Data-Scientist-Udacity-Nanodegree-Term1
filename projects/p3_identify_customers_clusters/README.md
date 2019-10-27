# Data Scientist Nanodegree
# Unsupervised Learning
## Project: Identify Customer Segments

### Installations

This project requires **Python 3.x** and the following Python libraries installed:

- scikit-learn==0.21.2
- pandas==0.24.2
- numpy==1.16.4
- matplotlib==3.1.0
- seaborn==0.9.0

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project.

### Project Motivation

As per Udacity Data Scientist Nanodegree project Term 1. This project is about applying  unsupervised learning techniques to identify segments of the population that form the core customer base for a mail-order sales company in Germany. These segments can then be used to direct marketing campaigns towards audiences that will have the highest expected rate of returns. The data used is provided by Bertelsmann Arvato Analytics, and represents a real-life data science task.


### summary of the project

In this notebook I wanted to analyze the Aravto customers. this project included using PCA to reduce the features set and KMeans clustering to identify the over presented and underpresented segmenets.

**Comparing the over and under presented groups we notice that:**
	- LP_STATUS_GROB_1.0: low income earners are not present in any of the under or over represented groups.
	
	- HH_EINKOMMEN_SCORE: under has more lower income compared to the presence of the highest income in the over represented.
	
	- PLZ8_ANTG3: under-represented had more customers with a lower share of 6-10 family homes compared to the average share home in over-represented group.
	
	- wealth: the under has more Prosperous Households compared to more wealthy households prescence in the over represented.
	
	- family_home_PLZ8: mainly 1-2 family homes in PLZ8 are present in both of these groups.
	
	- ALTERSKATEGORIE_GROB: the under re-presented contain more of the 46 - 60 years old compared to age groups of more than 60 years old in the over represented group.
	
	- FINANZ_VORSORGER: low prescence of be prepared Financial typology in the under-represented compared to the very low in over-represented.
	
	- ZABEOTYP_3: both groups did not have fair supply energy consumption.
	
	- SEMIO_ERL: the under-represented group had average event oriented personality compared to a high affinity in the over-represented group.
	
	- RETOURTYP_BK_S: both determined Minimal-Returner
	
	- SEMIO_VERT: under had dreamful with low affinity compared to the lowest affinity in the over-represented group.
	
	- SEMIO_SOZ: socially-minded had a very high affinity in the under group compared to the very low affinity in the over group.
	
	- SEMIO_FAM: family oriented had a low affinity compared to average affinity in the over group.
	
	- SEMIO_KULT: cultural-minded had the low affinity compared to average affinity in the over group.
	
	- FINANZTYP_5: both had a similar prescene of no investors.

Can we describe segments of the population that are relatively popular with the mail-order company, or relatively unpopular with the company?

- popular include the highest income people, age group above 60 years, wealthy households, event-oriented individuals including family and culutral minded groups with more of traditional attributes direction.

- unpopular include lower age groups, less prescence of low income and less affluent households and in terms of personality socially minded, low family and cultural minded groups are under presented.


### File Descriptions

- Identify_Customer_Segments.ipynb:a notebook containing the analysis for the data.

### Run

In a terminal or command window, navigate to the top-level project directory `Write_BlogPost/` (that contains this README) and run one of the following commands:

```bash
ipython notebook Identify_Customer_Segments.ipynb
```  
or
```bash
jupyter Identify_Customer_Segments.ipynb
```

This will open the iPython Notebook software and project file in your browser.

### Data

The data was provided by Bertelsmann Arvato Analytics.

### Licensing, Authors, Acknowledgements 

- This project is part of Data scientist Nanodegree from udacity 

This work is licensed under a [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/). Please refer to [Udacity Terms of Service](https://www.udacity.com/legal) for further information.
