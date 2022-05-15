# Analyze-A-B-Test-Results-Data-Analysis

Analyze A/B Test Results
This project will assure you have mastered the subjects covered in the statistics lessons. We have organized the current notebook into the following sections:

Introduction
Part I - Probability
Part II - A/B Test
Part III - Regression
Final Check
Submission
Specific programming tasks are marked with a ToDo tag.


Introduction
A/B tests are very commonly performed by data analysts and data scientists. For this project, you will be working to understand the results of an A/B test run by an e-commerce website. Your goal is to work through this notebook to help the company understand if they should:

Implement the new webpage,
Keep the old webpage, or
Perhaps run the experiment longer to make their decision.
Each ToDo task below has an associated quiz present in the classroom. Though the classroom quizzes are not necessary to complete the project, they help ensure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the rubric specification.

Tip: Though it's not a mandate, students can attempt the classroom quizzes to ensure statistical numeric values are calculated correctly in many cases.


Part I - Probability
To get started, let's import our libraries.

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
%matplotlib inline
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)
ToDo 1.1
Now, read in the ab_data.csv data. Store it in df. Below is the description of the data, there are a total of 5 columns:

Data columns	Purpose	Valid values
user_id	Unique ID	Int64 values
timestamp	Time stamp when the user visited the webpage	-
group	In the current A/B experiment, the users are categorized into two broad groups.
The control group users are expected to be served with old_page; and treatment group users are matched with the new_page.
However, some inaccurate rows are present in the initial data, such as a control group user is matched with a new_page.	['control', 'treatment']
landing_page	It denotes whether the user visited the old or new webpage.	['old_page', 'new_page']
converted	It denotes whether the user decided to pay for the company's product. Here, 1 means yes, the user bought the product.	[0, 1]
</center> Use your dataframe to answer the questions in Quiz 1 of the classroom.

Tip: Please save your work regularly.

a. Read in the dataset from the ab_data.csv file and take a look at the top few rows here:

df = pd.read_csv('ab_data.csv')
df.head()
user_id	timestamp	group	landing_page	converted
0	851104	2017-01-21 22:11:48.556739	control	old_page	0
1	804228	2017-01-12 08:01:45.159739	control	old_page	0
2	661590	2017-01-11 16:55:06.154213	treatment	new_page	0
3	853541	2017-01-08 18:28:03.143765	treatment	new_page	0
4	864975	2017-01-21 01:52:26.210827	control	old_page	1
b. Use the cell below to find the number of rows in the dataset.

df.shape
(294478, 5)
c. The number of unique users in the dataset.

df.user_id.nunique()
290584
d. The proportion of users converted.

df['converted'].sum() / len(df)
0.11965919355605512
e. The number of times when the "group" is treatment but "landing_page" is not a new_page.

len(df.query("(group == 'control') and (landing_page == 'new_page')") + df.query("(group == 'treatment') and\
                                                                                 (landing_page == 'old_page')"))
3893
f. Do any of the rows have missing values?

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 294478 entries, 0 to 294477
Data columns (total 5 columns):
user_id         294478 non-null int64
timestamp       294478 non-null object
group           294478 non-null object
landing_page    294478 non-null object
converted       294478 non-null int64
dtypes: int64(2), object(3)
memory usage: 11.2+ MB
ToDo 1.2
In a particular row, the group and landing_page columns should have either of the following acceptable values:

user_id	timestamp	group	landing_page	converted
XXXX	XXXX	control	old_page	X
XXXX	XXXX	treatment	new_page	X
It means, the control group users should match with old_page; and treatment group users should matched with the new_page.

However, for the rows where treatment does not match with new_page or control does not match with old_page, we cannot be sure if such rows truly received the new or old wepage.

Use Quiz 2 in the classroom to figure out how should we handle the rows where the group and landing_page columns don't match?

a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz. Store your new dataframe in df2.

df2 = df.drop(df[((df.group == 'control') & (df.landing_page == 'new_page')) | \
                 ((df.group == 'treatment') & (df.landing_page == 'old_page'))].index)
# Double Check all of the incorrect rows were removed from df2 - 
# Output of the statement below should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]
0
ToDo 1.3
Use df2 and the cells below to answer questions for Quiz 3 in the classroom.

a. How many unique user_ids are in df2?

df2.user_id.nunique()
290584
b. There is one user_id repeated in df2. What is it?

duplicate_user = df2[df2['user_id'].duplicated()].user_id
duplicate_user
2893    773192
Name: user_id, dtype: int64
c. Display the rows for the duplicate user_id?

df2[df2['user_id'] == duplicate_user.iloc[0]]
user_id	timestamp	group	landing_page	converted
1899	773192	2017-01-09 05:37:58.781806	treatment	new_page	0
2893	773192	2017-01-14 02:55:59.590927	treatment	new_page	0
d. Remove one of the rows with a duplicate user_id, from the df2 dataframe.

# Remove one of the rows with a duplicate user_id..
# Hint: The dataframe.drop_duplicates() may not work in this case because the rows with duplicate user_id are not entirely identical. 
df2.drop_duplicates(['user_id'], inplace=True)
# Check again if the row with a duplicate user_id is deleted or not
df2.shape
(290584, 5)
ToDo 1.4
Use df2 in the cells below to answer the quiz questions related to Quiz 4 in the classroom.

a. What is the probability of an individual converting regardless of the page they receive?


Tip: The probability you'll compute represents the overall "converted" success rate in the population and you may call it  ppopulation .

df2['converted'].sum() / len(df2)
0.11959708724499628
b. Given that an individual was in the control group, what is the probability they converted?

control_conversion = df2[df2['group'] == 'control']['converted'].sum() / len(df2[df2['group'] == 'control'])
control_conversion
0.1203863045004612
c. Given that an individual was in the treatment group, what is the probability they converted?

treatment_conversion = df2[df2['group'] == 'treatment']['converted'].sum() / len(df2[df2['group'] == 'treatment'])
treatment_conversion
0.11880806551510564
Tip: The probabilities you've computed in the points (b). and (c). above can also be treated as conversion rate. Calculate the actual difference (obs_diff) between the conversion rates for the two groups. You will need that later.

# Calculate the actual difference (obs_diff) between the conversion rates for the two groups.
d. What is the probability that an individual received the new page?

df2[df2['landing_page'] == 'new_page']['group'].count() / len(df2)
0.50006194422266881
e. Consider your results from parts (a) through (d) above, and explain below whether the new treatment group users lead to more conversions.

obs_diff = treatment_conversion - control_conversion
obs_diff
-0.0015782389853555567

Part II - A/B Test
Since a timestamp is associated with each event, you could run a hypothesis test continuously as long as you observe the events.

However, then the hard questions would be:

Do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?
How long do you run to render a decision that neither page is better than another?
These questions are the difficult parts associated with A/B tests in general.

ToDo 2.1
For now, consider you need to make the decision just based on all the data provided.

Recall that you just calculated that the "converted" probability (or rate) for the old page is slightly higher than that of the new page (ToDo 1.4.c).

If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should be your null and alternative hypotheses ( H0  and  H1 )?

You can state your hypothesis in terms of words or in terms of  pold  and  pnew , which are the "converted" probability (or rate) for the old and new pages respectively.

H0:pnew−pold≤0

H1:pnew−pold>0

ToDo 2.2 - Null Hypothesis  H0  Testing
Under the null hypothesis  H0 , assume that  pnew  and  pold  are equal. Furthermore, assume that  pnew  and  pold  both are equal to the converted success rate in the df2 data regardless of the page. So, our assumption is:


pnew  =  pold  =  ppopulation 
In this section, you will:

Simulate (bootstrap) sample data set for both groups, and compute the "converted" probability  p  for those samples.
Use a sample size for each group equal to the ones in the df2 data.
Compute the difference in the "converted" probability for the two samples above.
Perform the sampling distribution for the "difference in the converted probability" between the two simulated-samples over 10,000 iterations; and calculate an estimate.
Use the cells below to provide the necessary parts of this simulation. You can use Quiz 5 in the classroom to make sure you are on the right track.

a. What is the conversion rate for  pnew  under the null hypothesis?

p_new = df2['converted'].sum() / len(df2)
p_new
0.11959708724499628
b. What is the conversion rate for  pold  under the null hypothesis?

p_old = df2['converted'].sum() / len(df2)
p_old
0.11959708724499628
c. What is  nnew , the number of individuals in the treatment group?

Hint: The treatment group users are shown the new page.

n_new = df2[df2['landing_page'] == 'new_page']['landing_page'].count()
n_new
145310
d. What is  nold , the number of individuals in the control group?

n_old = df2[df2['landing_page'] == 'old_page']['landing_page'].count()
n_old
145274
e. Simulate Sample for the treatment Group
Simulate  nnew  transactions with a conversion rate of  pnew  under the null hypothesis.

Hint: Use numpy.random.choice() method to randomly generate  nnew  number of values.
Store these  nnew  1's and 0's in the new_page_converted numpy array.

treatment_df = df2.query('group == "treatment"')
sample_new = treatment_df.sample(n_new, replace=True)
new_page_converted = sample_new['converted']
new_page_converted.mean()
0.11799600853348015
f. Simulate Sample for the control Group
Simulate  nold  transactions with a conversion rate of  pold  under the null hypothesis.
Store these  nold  1's and 0's in the old_page_converted numpy array.

control_df = df2.query('group == "control"')
sample_old = control_df.sample(n_old, replace=True)
old_page_converted = sample_old['converted']
old_page_converted.mean()
0.11835565896168619
g. Find the difference in the "converted" probability  (p′new  -  p′old)  for your simulated samples from the parts (e) and (f) above.

p_diff_simulate = new_page_converted.mean() - old_page_converted.mean()
p_diff_simulate
-0.00035965042820604309
h. Sampling distribution
Re-create new_page_converted and old_page_converted and find the  (p′new  -  p′old)  value 10,000 times using the same simulation process you used in parts (a) through (g) above.


Store all  (p′new  -  p′old)  values in a NumPy array called p_diffs.

# Sampling distribution 
p_diffs = []

for _ in range(10000):
    new_page_converted = np.random.binomial(1,p_new,n_new).mean()
    old_page_converted = np.random.binomial(1,p_old,n_old).mean()
    p_diffs.append(new_page_converted - old_page_converted)
i. Histogram
Plot a histogram of the p_diffs. Does this plot look like what you expected? Use the matching problem in the classroom to assure you fully understand what was computed here.


Also, use plt.axvline() method to mark the actual difference observed in the df2 data (recall obs_diff), in the chart.

Tip: Display title, x-label, and y-label in the chart.

p_diffs = np.array(p_diffs)
plt.hist(p_diffs)
(array([   10.,    77.,   453.,  1494.,  2816.,  2961.,  1650.,   470.,
           66.,     3.]),
 array([ -4.86822351e-03,  -3.90051981e-03,  -2.93281612e-03,
         -1.96511242e-03,  -9.97408719e-04,  -2.97050205e-05,
          9.37998678e-04,   1.90570238e-03,   2.87340607e-03,
          3.84110977e-03,   4.80881347e-03]),
 <a list of 10 Patch objects>)

j. What proportion of the p_diffs are greater than the actual difference observed in the df2 data?

(p_diffs > obs_diff).mean()
0.89970000000000006
k. Please explain in words what you have just computed in part j above.

What is this value called in scientific studies?
What does this value signify in terms of whether or not there is a difference between the new and old pages? Hint: Compare the value above with the "Type I error rate (0.05)".
90,56% is the proportion of the p_diffs that are greater than the actual difference observed in ab_data.csv.

l. Using Built-in Methods for Hypothesis Testing
We could also use a built-in to achieve similar results. Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance.

Fill in the statements below to calculate the:

convert_old: number of conversions with the old_page
convert_new: number of conversions with the new_page
n_old: number of individuals who were shown the old_page
n_new: number of individuals who were shown the new_page
import statsmodels.api as sm

# number of conversions with the old_page
convert_old = len(df2.query('landing_page == "old_page" & converted == 1'))

# number of conversions with the new_page
convert_new = len(df2.query('landing_page == "new_page" & converted == 1'))

# number of individuals who were shown the old_page
n_old = len(df2.query('landing_page == "old_page"'))
# number of individuals who received new_page
n_new =len(df2.query('landing_page == "new_page"'))
m. Now use sm.stats.proportions_ztest() to compute your test statistic and p-value. Here is a helpful link on using the built in.

The syntax is:

proportions_ztest(count_array, nobs_array, alternative='larger')
where,

count_array = represents the number of "converted" for each group
nobs_array = represents the total number of observations (rows) in each group
alternative = choose one of the values from [‘two-sided’, ‘smaller’, ‘larger’] depending upon two-tailed, left-tailed, or right-tailed respectively.
Hint:
It's a two-tailed if you defined  H1  as  (pnew=pold) .
It's a left-tailed if you defined  H1  as  (pnew<pold) .
It's a right-tailed if you defined  H1  as  (pnew>pold) .

The built-in function above will return the z_score, p_value.

About the two-sample z-test
Recall that you have plotted a distribution p_diffs representing the difference in the "converted" probability  (p′new−p′old)  for your two simulated samples 10,000 times.

Another way for comparing the mean of two independent and normal distribution is a two-sample z-test. You can perform the Z-test to calculate the Z_score, as shown in the equation below:

Zscore=(p′new−p′old)−(pnew−pold)σ2newnnew+σ2oldnold−−−−−−−−√
 
where,

p′  is the "converted" success rate in the sample
pnew  and  pold  are the "converted" success rate for the two groups in the population.
σnew  and  σnew  are the standard deviation for the two groups in the population.
nnew  and  nold  represent the size of the two groups or samples (it's same in our case)
Z-test is performed when the sample size is large, and the population variance is known. The z-score represents the distance between the two "converted" success rates in terms of the standard error.

Next step is to make a decision to reject or fail to reject the null hypothesis based on comparing these two values:

Zscore 
Zα  or  Z0.05 , also known as critical value at 95% confidence interval.  Z0.05  is 1.645 for one-tailed tests, and 1.960 for two-tailed test. You can determine the  Zα  from the z-table manually.
Decide if your hypothesis is either a two-tailed, left-tailed, or right-tailed test. Accordingly, reject OR fail to reject the null based on the comparison between  Zscore  and  Zα .

Hint:
For a right-tailed test, reject null if  Zscore  >  Zα .
For a left-tailed test, reject null if  Zscore  <  Zα .

In other words, we determine whether or not the  Zscore  lies in the "rejection region" in the distribution. A "rejection region" is an interval where the null hypothesis is rejected iff the  Zscore  lies in that region.

Reference:

Example 9.1.2 on this page/09%3A_Two-Sample_Problems/9.01%3A_Comparison_of_Two_Population_Means-_Large_Independent_Samples), courtesy www.stats.libretexts.org
Tip: You don't have to dive deeper into z-test for this exercise. Try having an overview of what does z-score signify in general.

# ToDo: Complete the sm.stats.proportions_ztest() method arguments
z_score, p_value =sm.stats.proportions_ztest([convert_old, convert_new], [n_old, n_new],value=None, alternative='smaller', prop_var=False)
print(z_score, p_value)
1.31092419842 0.905058312759
n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages? Do they agree with the findings in parts j. and k.?


Tip: Notice whether the p-value is similar to the one computed earlier. Accordingly, can you reject/fail to reject the null hypothesis? It is important to correctly interpret the test statistic and p-value.

The p_value is 0.9 and is higher than 0.05 significance level. That means we can not be confident with a 95% confidence level that the converted rate of the new_page is larger than the old_page.


Part III - A regression approach
ToDo 3.1
In this final part, you will see that the result you achieved in the A/B test in Part II above can also be achieved by performing regression.


a. Since each row in the df2 data is either a conversion or no conversion, what type of regression should you be performing in this case?

Logistic regression

b. The goal is to use statsmodels library to fit the regression model you specified in part a. above to see if there is a significant difference in conversion based on the page-type a customer receives. However, you first need to create the following two columns in the df2 dataframe:

intercept - It should be 1 in the entire column.
ab_page - It's a dummy variable column, having a value 1 when an individual receives the treatment, otherwise 0.
from scipy import stats
df2['intercept'] = 1
df2['ab_page'] = pd.get_dummies(df2['group'])['treatment']
df2.head()
user_id	timestamp	group	landing_page	converted	intercept	ab_page
0	851104	2017-01-21 22:11:48.556739	control	old_page	0	1	0
1	804228	2017-01-12 08:01:45.159739	control	old_page	0	1	0
2	661590	2017-01-11 16:55:06.154213	treatment	new_page	0	1	1
3	853541	2017-01-08 18:28:03.143765	treatment	new_page	0	1	1
4	864975	2017-01-21 01:52:26.210827	control	old_page	1	1	0
c. Use statsmodels to instantiate your regression model on the two columns you created in part (b). above, then fit the model to predict whether or not an individual converts.

log_mod = sm.Logit(df2['converted'], df2[['intercept', 'ab_page']])
results = log_mod.fit()
Optimization terminated successfully.
         Current function value: 0.366118
         Iterations 6
d. Provide the summary of your model below, and use it as necessary to answer the following questions.

results.summary()
Logit Regression Results
Dep. Variable:	converted	No. Observations:	290584
Model:	Logit	Df Residuals:	290582
Method:	MLE	Df Model:	1
Date:	Wed, 20 Apr 2022	Pseudo R-squ.:	8.077e-06
Time:	15:19:19	Log-Likelihood:	-1.0639e+05
converged:	True	LL-Null:	-1.0639e+05
LLR p-value:	0.1899
coef	std err	z	P>|z|	[0.025	0.975]
intercept	-1.9888	0.008	-246.669	0.000	-2.005	-1.973
ab_page	-0.0150	0.011	-1.311	0.190	-0.037	0.007
e. What is the p-value associated with ab_page? Why does it differ from the value you found in Part II?


Hints:

What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in Part II?
You may comment on if these hypothesis (Part II vs. Part III) are one-sided or two-sided.
You may also compare the current p-value with the Type I error rate (0.05).
The p-value associated with ab_page is 0.19. The null cannot be rejected because 0.19 is above our Type I error threshold of 0.05. the conversion rate of the old_page is less than the conversion rate of the new_page. the conversion rate of the old_page is less than the conversion rate of the new_page. This assumes a one-tailed test. In Part III, the alternative hypothesis can be formulated as follows The landing_page type influences the conversion rate or the conversion rate of the old_page is different to the conversion rate of the new_page. This assumes a two-tailed test.

f. Now, you are considering other things that might influence whether or not an individual converts. Discuss why it is a good idea to consider other factors to add into your regression model. Are there any disadvantages to adding additional terms into your regression model?

It is a good idea to consider other factors in order to identify other potencial influences on the conversion rate and the disadvantage is that the model gets more complex

g. Adding countries
Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives in.

You will need to read in the countries.csv dataset and merge together your df2 datasets on the appropriate rows. You call the resulting dataframe df_merged. Here are the docs for joining tables.

Does it appear that country had an impact on conversion? To answer this question, consider the three unique values, ['UK', 'US', 'CA'], in the country column. Create dummy variables for these country columns.

Hint: Use pandas.get_dummies() to create dummy variables. You will utilize two columns for the three dummy variables.

Provide the statistical output as well as a written response to answer this question.

countries_df = pd.read_csv('./countries.csv')
df_new = countries_df.set_index('user_id').join(df2.set_index('user_id'), how='inner')
df_new.head()
country	timestamp	group	landing_page	converted	intercept	ab_page
user_id							
834778	UK	2017-01-14 23:08:43.304998	control	old_page	0	1	0
928468	US	2017-01-23 14:44:16.387854	treatment	new_page	0	1	1
822059	UK	2017-01-16 14:04:14.719771	treatment	new_page	1	1	1
711597	UK	2017-01-22 03:14:24.763511	control	old_page	0	1	0
710616	UK	2017-01-16 13:14:44.000513	treatment	new_page	0	1	1
dum_countries = pd.get_dummies(df_new['country'])
df4 = dum_countries.join(df_new, how='inner')
df4.head()
CA	UK	US	country	timestamp	group	landing_page	converted	intercept	ab_page
user_id										
834778	0	1	0	UK	2017-01-14 23:08:43.304998	control	old_page	0	1	0
928468	0	0	1	US	2017-01-23 14:44:16.387854	treatment	new_page	0	1	1
822059	0	1	0	UK	2017-01-16 14:04:14.719771	treatment	new_page	1	1	1
711597	0	1	0	UK	2017-01-22 03:14:24.763511	control	old_page	0	1	0
710616	0	1	0	UK	2017-01-16 13:14:44.000513	treatment	new_page	0	1	1
log_mod2 = sm.Logit(df4['converted'], df4[['intercept', 'ab_page', 'UK', 'CA']])
results = log_mod2.fit()
results.summary()
Optimization terminated successfully.
         Current function value: 0.366113
         Iterations 6
Logit Regression Results
Dep. Variable:	converted	No. Observations:	290584
Model:	Logit	Df Residuals:	290580
Method:	MLE	Df Model:	3
Date:	Wed, 20 Apr 2022	Pseudo R-squ.:	2.323e-05
Time:	15:31:55	Log-Likelihood:	-1.0639e+05
converged:	True	LL-Null:	-1.0639e+05
LLR p-value:	0.1760
coef	std err	z	P>|z|	[0.025	0.975]
intercept	-1.9893	0.009	-223.763	0.000	-2.007	-1.972
ab_page	-0.0149	0.011	-1.307	0.191	-0.037	0.007
UK	0.0099	0.013	0.743	0.457	-0.016	0.036
CA	-0.0408	0.027	-1.516	0.130	-0.093	0.012
h. Fit your model and obtain the results
Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if are there significant effects on conversion. Create the necessary additional columns, and fit the new model.

Provide the summary results (statistical output), and your conclusions (written response) based on the results.

Tip: Conclusions should include both statistical reasoning, and practical reasoning for the situation.

Hints:

Look at all of p-values in the summary, and compare against the Type I error rate (0.05).
Can you reject/fail to reject the null hypotheses (regression model)?
Comment on the effect of page and country to predict the conversion.
df_new.groupby(['country','ab_page'], as_index=False).mean()
country	ab_page	converted	intercept
0	CA	0	0.118783	1.0
1	CA	1	0.111902	1.0
2	UK	0	0.120022	1.0
3	UK	1	0.121171	1.0
4	US	0	0.120630	1.0
5	US	1	0.118466	1.0
df_new['intercept'] = 1

lm = sm.Logit(df_new['converted'],df_new[['intercept','ab_page','US','interaction_us_ab_page','CA','interaction_ca_ab_page']])
results = lm.fit()
results.summary()
 
there is not enough evidence that the new_page increases the conversion rate as compared to the old_page. This is based on the probability figures, A/B testand regression. There is no strong evidence that the countries (US, CA and UK) influence the conversion rate. Since the sample size is large continuing the testing of the new_page is likely not necessary. It is best to focus on the development of another new landing page.


Final Check!
Congratulations! You have reached the end of the A/B Test Results project! You should be very proud of all you have accomplished!

Tip: Once you are satisfied with your work here, check over your notebook to make sure that it satisfies all the specifications mentioned in the rubric. You should also probably remove all of the "Hints" and "Tips" like this one so that the presentation is as polished as possible.


Submission
You may either submit your notebook through the "SUBMIT PROJECT" button at the bottom of this workspace, or you may work from your local machine and submit on the last page of this project lesson.

Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).
Alternatively, you can download this report as .html via the File > Download as submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.
Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations!
from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])
