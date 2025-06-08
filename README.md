# Forex Trading Pair Forecaster
---
![Demo](/Report/demo.gif)

--- 
### Introduction
This prototype presents a deployed machine learning application designed to address the critical question of when to trade currencies. In light of recent global challenges—such as the recent international economic conflicts with the COVID-19 pandemic and the resulting hyperinflation—currency values have become increasingly volatile and unpredictable. To mitigate this uncertainty, we propose a predictive model that forecasts daily exchange rates for major currency pairs. For this prototype, we focus on the three most actively traded pairs in the forex market: EUR/USD, JPY/USD, and GBP/USD.

![ETL](/Report/ETLpipeline.png)
We utilized AWS to host our backend scraping, ETL pipeline, machine learning models and front-end streamlit app. The resources we used are 
1. t2.micro for webscraping on FXStreet.com
2. t3.large for hosting our machine learning models 
3. t2.micro to host our streamlit application. 
4. S3 Buckets to host .parquet dataframe files

### Machine Learning Models
**finBERT** from HuggingFace was used in financial-text sentiment analysis.
**Prophet** from Facebook/Meta was used in time-series forecasting post-feature engineering 

----
### Future Steps 
- [ ] Use of server-less compute on AWS, via lambda functions 
- [ ] Use of Amazon SageMaker for serverless-hosting 
- [ ] Transition to use Relational Databases instead of S3 for faster, and more structured data retrieval 
- [ ] Use LLMs in financial text analysis for more optimized/ context-understanding sentiment score generation. 
- [ ] Add a 'strategies' plan for predicted best currencies to exchange 
---
### How to Run this App
Locally: 
1. Initialize your own virtual env within the project `python -m venv venv` 
2. Install dependencies `pip install requirements.txt`
3. Run the app! `streamlit run app.py`

In AWS: 
*in-progress*

### How to Use this App
---
The app for now simply acts like a dashboard. The app will indicate it's suggested action per each trading pair as listed in Yahoo Finance. 

The user may toggle forecast result to include/exclude Forex news data. This is because the lack of accurate sentiment extraction introduced noise to the model, which increased RMSE (Root-Mean-Square Error) across all forecasts.

### Stack/ Dependencies

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1869px-Python-logo-notext.svg.png" width="100px"/> <img src="https://img.icons8.com/color/512/pandas.png" width="100px"/> <img src="https://images.crunchbase.com/image/upload/c_pad,f_auto,q_auto:eco,dpr_1/vgay5hqdvszlmvud3hwu" width="100px"/> <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTYDvVvSTSxKKvBwZAf9c9UWMY2yOfZvPq46g&s" width="100px"/> <img src="https://static-00.iconduck.com/assets.00/pytorch-icon-1694x2048-jgwjy3ne.png" width="100px"/> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/1200px-Scikit_learn_logo_small.svg.png" width="100px"/> <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR86kDWkquaiBSCj1nHaJTsCTNlVPH0GR4H2w&s
" width="100px"/> <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTGDKmSgL7UJ6sstMUQTtjI2iDN7ClN2jRZ5Q&s
" width="100px"/> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/Amazon_Web_Services_Logo.svg/1200px-Amazon_Web_Services_Logo.svg.png
" width="100px"/> <img src="https://static-00.iconduck.com/assets.00/aws-ec2-icon-1696x2048-nhw31ife.png
" width="100px"/> <img src="https://static-00.iconduck.com/assets.00/aws-s3-simple-storage-service-icon-1694x2048-ygs8j98c.png
" width="100px"/> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Jupyter_logo.svg/1200px-Jupyter_logo.svg.png
" width="100px"/> <img src="https://miro.medium.com/v2/resize:fit:1400/0*GUKP-h4wWRK5k-cQ.png
" width="100px"/> 

