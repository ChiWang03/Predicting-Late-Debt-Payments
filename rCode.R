library(ggplot2)

df <- read.csv('SCFP2013S')

#Add another variable called PCTHPAID, which is HOMEEQ/HOUSES 
df$PCTHPAID<- df$HOMEEQ/df$HOUSES 
df$PCTHPAID[is.nan(df$PCTHPAID)] <- 0

df1 <- df[,c('LATE60', 'EDCL', 'KIDS', 'RACE', 'WSAVED', 'EXPENSHILO', 'NORMINC', 'TURNDOWN', 'EQUITINC', 'HOTHFIN','PCTHPAID','DEBT2INC')]

#Transforming variables 
#plot to see the distribution of data
plot(factor(df1$LATE60), df1$NORMINC)
summary(df1$NORMINC)
#NORMINC 
df1$NORMINC[df1$NORMINC==0] <- 1
#Now we log transform NORMINC
df1$lnNORMINC <- log(df1$NORMINC+150)

#plot lnNORMINC, which is NORMINC after transformation 
plot(factor(df1$LATE60), df1$lnNORMINC)
summary(df1$lnNORMINC)

#EQUITINC
plot(factor(df1$LATE60), df1$EQUITINC)
summary(df1$EQUITINC)
#We take root 10 to normalize the data better (because of the huge MAX value)
df1$rtEQUITINC <- (df1$EQUITINC)**(1/10)

plot(factor(df1$LATE60),(df1$rtEQUITINC))
summary((df1$rtEQUITINC))

plot(factor(df1$LATE60), df1$DEBT2INC)
summary(df1$DEBT2INC)
#For every DEBT2INC == 10 we set it to the power of 10 and take the log with base 10 of the data
#the reason why we keep it 10 is because the debt to income ratio is high and automatically set to 10 based on
#the explanation in the original data set
df1$DEBT2INC[df1$DEBT2INC==10] <-10^10
df1$DEBT2INC[df1$DEBT2INC==0] <- 1

#log transform DEBT2INC
df1$log10DEBT2INC <- log10(df1$DEBT2INC)

plot(factor(df1$LATE60), df1$log10DEBT2INC)
summary(df1$log10DEBT2INC)

#Combining individuals who have 4 or more kids to be just 4 kids 
#since there are only a few individuals who have more than 4 kids 

df1$KIDS[df1$KIDS==0]<-0
df1$KIDS[df1$KIDS==1]<-1
df1$KIDS[df1$KIDS==2]<-2
df1$KIDS[df1$KIDS==3]<-3
df1$KIDS[df1$KIDS==4]<-4
df1$KIDS[df1$KIDS==5]<-4
df1$KIDS[df1$KIDS==6]<-4
df1$KIDS[df1$KIDS==7]<-4
df1$KIDS[df1$KIDS==8]<-4

#We see that the individual that have more than 4 kids are all merged under 4 kids
as.data.frame(table(df1$KIDS))

plot(factor(df1$LATE60), df1$PCTHPAID)
summary(df1$PCTHPAID)

#We do an exponential transform since PCTHPAID contains negative values 
df1$expPCTHPAID <- exp(df1$PCTHPAID)

#plot after transform
plot(factor(df1$LATE60), df1$expPCTHPAID)
summary(df1$expPCTHPAID)

#Factor the categorical variables
df1$EDCL <- factor(df1$EDCL);
df1$KIDS <- factor(df1$KIDS);
df1$RACE <- factor(df1$RACE);
df1$WSAVED <- factor(df1$WSAVED);
df1$EXPENSHILO <- factor(df1$EXPENSHILO);
df1$TURNDWON <- factor(df1$TURNDOWN);
df1$HOTHIN <- factor(df1$HOTHFIN);

df2 <- df1[,c('LATE60','EDCL', 'KIDS', 'RACE', 'WSAVED', 'EXPENSHILO', 'lnNORMINC', 'TURNDOWN', 'rtEQUITINC', 'HOTHFIN','expPCTHPAID','log10DEBT2INC')]
names(df2)

model1 <- glm(LATE60~EDCL+KIDS+RACE+WSAVED+EXPENSHILO+lnNORMINC+TURNDOWN+HOTHFIN+expPCTHPAID+log10DEBT2INC+rtEQUITINC, family = "binomial", data = df2)
summary(model1)

set.seed(12345)
# 4000 for training set, 6015-4000 = 2015 for holdout
ntot <- nrow(df1) #number of rows
iperm<-sample(ntot,ntot) # random permutation of 1...ntot
n<-4000
train<-df2[iperm[1:n],]
hold<-df2[iperm[(n+1):ntot],]

cat('The total # of YES in the training set is',sum(train$LATE60),', representing',mean(train$LATE60)*100,"%")

attach(train)
train_matrix_form <- data.matrix(train)
cor(train_matrix_form[,1],train_matrix_form[,c(2:12)])
detach(train)

#Getting full and null model for stepwise regression.
full.model <- glm(LATE60~EDCL+KIDS+RACE+WSAVED+EXPENSHILO+lnNORMINC+TURNDOWN+HOTHFIN+expPCTHPAID+log10DEBT2INC+rtEQUITINC, family = "binomial", data = train)
null.model <- glm(LATE60~1,family="binomial", data = train) 

# Stepwise Regression.
fit <-step(null.model, scope=list(lower=null.model, upper=full.model),direction="both")

#fit1 is the the best AIC model
fit1 <- glm(LATE60 ~ expPCTHPAID + WSAVED + TURNDOWN + EDCL + KIDS + rtEQUITINC + EXPENSHILO + log10DEBT2INC, family = "binomial", data = train)
summary(fit1,EDCL=3)

#fit2 model excludes log10DEBT2INC
#fit2 has residual deviance 1442.2 and AIC 1472
fit2 <-glm(LATE60~expPCTHPAID + WSAVED + TURNDOWN + EDCL + KIDS + rtEQUITINC + EXPENSHILO, family = "binomial", data = train)
summary(fit2)

#fit3 model the orignal model 
#fit 3 has residual deviance of 1434.0 and AIC 1476
fit3 <- full.model
summary(fit3)
#Correlation table
df1_matrix_form <- data.matrix(df1)
cor(df1_matrix_form[,1],df1_matrix_form[,12])

#Missclassification on fit1 and fit2 to find better model
#Predict the probabilities of fit1 and fit2

#fit1
pred1 <- predict(fit1,type='response');
head(pred1)
pred1[1] #is the probabilty that #4337 in the training dataset has late payment using fit1
summary(pred1)

#fit2
pred2 <- predict(fit2, type = 'response');
head(pred2)
pred2[1] # The probablity predicted that #4337 has late payment using fit2
summary(pred2)

#The summary of pred1 and pred2 just tells us the prediction probabilities of a person defaulting between
#2 models fit1 and fit2

tab1a<-table(train$LATE60,pred1>0.5)
tab1b<-table(train$LATE60,pred1>0.3)
tab1c<-table(train$LATE60,pred1>0.1)
tab2a<-table(train$LATE60,pred2>0.5)
tab2b<-table(train$LATE60,pred2>0.3)
tab2c<-table(train$LATE60,pred2>0.1)

tab1a; tab1b; tab1c;
tab2a; tab2b; tab2c;

#compare fit1 and fit2 missiclassification with probabilities over 0.5
print(tab1a/apply(tab1a,1,sum));print(tab2a/apply(tab2a,1,sum));


#compare fit1 and fit2 missiclassification with probabilities over 0.3
print(tab1b/apply(tab1b,1,sum));print(tab2b/apply(tab2b,1,sum));


#compare fit1 and fit2 missiclassification with probabilities over 0.1
print(tab1c/apply(tab1b,1,sum));print(tab2c/apply(tab2c,1,sum));


#again we look at the prediction probabilities
pred1.hold<-predict(fit1,type="response",newdata=hold)
summary(pred1.hold);
pred2.hold<-predict(fit2,type="response",newdata=hold)
summary(pred2.hold);

htab1a<-table(hold$LATE60,pred1.hold>0.5)
htab1b<-table(hold$LATE60,pred1.hold>0.3)
htab1c<-table(hold$LATE60,pred1.hold>0.1)
htab2a<-table(hold$LATE60,pred2.hold>0.5)
htab2b<-table(hold$LATE60,pred2.hold>0.3)
htab2c<-table(hold$LATE60,pred2.hold>0.1)

htab1a; htab1b; htab1c;
htab2a; htab2b; htab2c;

#compare fit1 and fit2 missiclassification with probabilities over 0.5
print(htab1a/apply(htab1a,1,sum));   print(htab2a/apply(htab2a,1,sum));

#compare fit1 and fit2 missiclassification with probabilities over 0.3
print(htab1b/apply(htab1b,1,sum));   print(htab2b/apply(htab2b,1,sum));

#compare fit1 and fit2 missiclassification with probabilities over 0.1
print(htab1c/apply(htab1c,1,sum));   print(htab2c/apply(htab2c,1,sum));

###-####
#Visualizing Frequency Tables
###-###

#EDCL
yesEDCL <- c(0.0588235, 0.083697, 0.0972762, 0.0351864)*100
catEDCL <- c(1, 2, 3, 4)
e=data.frame(yesEDCL, catEDCL)
e$catEDCL <- factor(e$catEDCL, labels=c("No High School", "High School", "Some college", "College degree"), level=1:4)
#c <- c("darkred","#0072B2","darkgreen","#D55E00")
edcl=ggplot(data=e ,aes(x=catEDCL, y=yesEDCL)) + geom_bar(colour="black",fill="#D55E00", stat="identity",width=.5) + 
labs(x="Education" ,y="Percentage(%)") + ggtitle("Education and Late Debt Payments")

#KIDS 
yesKIDS <- c(0.0421546, 0.085308, 0.0797814, 0.080188679, 0.12195121)*100
catKIDS <- c(1,2,3,4,5)
d=data.frame(yesKIDS,catKIDS)
d$catKIDS <- factor(d$catKIDS, labels=c("0 Kids", "1 Kid", "2 Kid", "3 Kids","4 or more kids"), level=1:5)
#c <- c("darkred","#0072B2","darkgreen","#D55E00","#9999CC")
kids=ggplot(data=d ,aes(x=catKIDS, y=yesKIDS)) + geom_bar(colour="black",fill="#D55E00", stat="identity",width=.5) + 
labs(x="Kids" ,y="Percentage(%)") + ggtitle("Kids and Late Debt Payments")

yesRACE <- c(0.0472529, 0.128514, 0.088288, 0.04137)*100
catRACE <- c(1,2,3,4)
#catRACE <- c("White","Black","Hispanic","Others")
k=data.frame(yesRACE,catRACE)
k$catRACE <- factor(k$catRACE, labels=c("White", "Black", "Hispanic", "Others"), level=1:4)
#c <- c("darkred","#0072B2","darkgreen","#D55E00")
race=ggplot(data=k ,aes(x=catRACE, y=yesRACE)) + geom_bar(colour="black",fill="#D55E00", stat="identity",width=.5) + 
labs(x="Race" ,y="Percentage(%)") + ggtitle("Race and Late Debt Payments")

#EXPENSHILO 
yesEXPENSHILO <- c(0.08608, 0.10526, 0.0458739)*100
catEXPENSHILO <- c(1,2,3)
ok=data.frame(yesEXPENSHILO,catEXPENSHILO)
ok$catEXPENSHILO <- factor(ok$catEXPENSHILO, labels=c("unusually high", "unusually low", "normal"), level=1:3)
#c <- c("darkred","#0072B2","darkgreen","#D55E00")
expen=ggplot(data=ok ,aes(x=catEXPENSHILO, y=yesEXPENSHILO)) + geom_bar(colour="black",fill="#D55E00", stat="identity",width=.5) + 
  labs(x="EXPENSHILO" ,y="Percentage(%)") + ggtitle("Household overall expenses and Late Debt Payments")

#WSAVED  
yesWSAVED <-  c(0.1644336, 0.080466, 0.02804262479)*100
catWSAVED <- c(1,2,3)
ws=data.frame(yesWSAVED,catWSAVED)
ws$catWSAVED <- factor(ws$catWSAVED, labels=c("Spending > Income", "Spending = Income", "Spending < Income"), level=1:3)
#c <- c("darkred","#0072B2","darkgreen","#D55E00")
wsave=ggplot(data=ws ,aes(x=catWSAVED, y=yesWSAVED)) + geom_bar(colour="black",fill="#D55E00", stat="identity",width=.5) + 
  labs(x="WSAVED" ,y="Percentage(%)") + ggtitle("Spending Habits and Late Debt Payments")

multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

multiplot(edcl, kids, race, expen, wsave, cols=2)

par(mfrow=c(2,2))
plot(factor(df1$LATE60), df1$lnNORMINC, main="LATE60 vs lnNORMINC"); 
plot(factor(df1$LATE60), df1$log10DEBT2INC, main="LATE60 vs log10DEBT2INC"); 
plot(factor(df1$LATE60), df1$rtEQUITINC, main="LATE60 vs rtEQUITINC"); 
plot(factor(df1$LATE60), df1$expPCTHPAID, main="LATE60 vs expPCTHPAID")
