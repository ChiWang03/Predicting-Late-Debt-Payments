# Predicting-Late-Debt-Payments

This mini paper on predicting late debt paymnts was written by Chi Wang and Matthias Lee, with the help of three other group members for statistics 306 of the University of British Columbia. The project looks into the data set: 2013 Survey of Consumer Finances (SCF), and identifies the charactristics that best predict people who have late debt payments using logistic regression. 









### Executive summary:

  The objective of this project is to identify the characteristics that best predict people who have late debt payments. As such, we decided to use the data from the 2013 Survey of Consumer Finances (SCF)1. The dataset contains a variable, LATE60 that denotes if a household had a debt payment 60 days or more past due within the last year and is used as the dependent variable. The 12 predictor variables chosen are based on what the team thinks are simple, generic questions that could be asked by a bank to quickly determine if a person will have late debt payments. The predictor variables include level of education, number of kids in a household, race, spending habits within the past year, overall household expenses within the last year, income, if the respondent or his/her spouse has been turned down for credit or received only partial credit by a lender within the last 5 years, if the household owns any financial assets (defined in the Appendix), debt-to- income ratio, ratio of equity to normal income, and percentage of a home that is paid off. Our final model uses 9 of the 12 predictor variables including: level of education, number of kids in the household, spending habits within the past year, percentage of a home that is paid off, if a household has been turned down by a creditor, ratio of equity to normal income, debt-to-income ratio.
  The odds ratio obtained from these variables support most of our initial presumptions. However, the model reveals a counter-intuitive aspect of two explanatory variables. Holding all other variables in the final model constant, a person with a college degree is more likely to have late debt payments relative to a person with no high school diploma. In addition, an individual who has a higher pencentage of a house paid off tends to have a higher odds of having late debt payments.
