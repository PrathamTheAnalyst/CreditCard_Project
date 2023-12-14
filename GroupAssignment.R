# WST 212 - Project

#-------------------------------------------------------------------
# DO NOT make any changes to the following code:

## Load necessary packages:
library(readr)
library(caret)
library(dplyr)
library(ggplot2)
library(naivebayes)
library(e1071)
library(rpart)
library(pROC)
install.packages("pROC")
install.packages("mlbench")
install.packages("naivebayes") 
install.packages("rpart")

## Import data
creditcard <- read.csv("creditcard.csv")


#-------------------------Explanatory data analysis-------------------------

ccsansTime<-creditcard[,2:31]

#Structure of data
(dimensions<-dim (creditcard))
(variables<-names(creditcard))

#check if we have any fraudulent events
any(creditcard$Class==1) # There are

#nature of data
str(creditcard)
glimpse(creditcard)
tail(creditcard)
summary(creditcard)


#Data visualisation 
#Shows the discrepancy in data by class
ggplot(data=creditcard)+geom_bar(aes(x=factor(creditcard$Class)))+
  labs(x="Credit Card Transaction Classification", y="Frequency", title="Frequency Comparison of Credit Card Transaction Classification")

#Information of exclusively fraudulent transactions
FTransact<-creditcard[creditcard$Class == 1, ]

#numerical proof of class imbalance
dim(FTransact)
dim(creditcard)



#------------------------------Modelling-------------------------------------


#Convert character variable to numeric  
# zero indicates fraudulent and one indicate non-fraudulent
creditcard$Class <- factor(ifelse(creditcard$Class == 1,0,1))

#Train/Test Split
#Preparing Training and Test data with a 70%/30% split
set.seed(123)
split <- round(nrow(creditcard)*0.70)
train  <- sample(1:nrow(creditcard), split, replace=FALSE)
TrainData <- creditcard[train, ]
TestData  <- creditcard[-train, ]


#---------------------Logistic Regression------------------------------------

# Logistic Regression Model(Based on all the variables in the creditcard dataset)

logistic_regmod <- glm(Class ~., 
                       family = "binomial", 
                       data=TrainData)

# Model Specifications(Statistically significant variables)
summary(logistic_regmod)

#Prediction on test set
prob <- predict(logistic_regmod, 
                newdata = TestData, 
                type = "response")

#Classification
threshold <- 0.25
y_pred <- ifelse(prob > 0.25, 1, 0)


# Convert the predicted response to a factor so that it can be compared with 
# the observed response in the test dataset.
y_predict <- factor(y_pred, levels=c(0, 1))
y_actual <- TestData$Class


# Obtaining a confusion matrix and performance measures
(CM <- confusionMatrix(data=y_predict, 
                       reference=TestData$Class,
                       positive = "1"))

precision <- CM$byClass["Precision"]
recall <- CM$byClass["Recall"]
specificity <- CM$byClass["Specificity"]

(F1 <- 2*(precision*recall)/(precision+recall))

#---------------------Naive Bayes classification-----------------------------


# naive bayes classification model
naive_model <- naiveBayes(Class ~ ., data = TrainData ) 

# predicting the model
predicted_values <- predict(naive_model, TestData)

#Evaluate the model
a_check <-table(predicted_values,TestData$Class)

# probabilities asscociated with a model
predicted_values_probabilities <- predict(naive_model, TestData, type="raw")

#CHECKING THE ACCURACY FOR BOTH TRAINDATA AND TESTDATA
check_pred <-table(predicted_values,TestData$Class)
check_test <- 1-sum(diag(check_pred ))/sum(check_pred ) 

# missclassification in testData is approximately 2.114711 %

# checking accuracy
accuracy <- 1-check_test
#  test data accuracy is approximately 97.88529 %

check_pred_train  <-predict(naive_model, TrainData)
check_train <- table( check_pred_train ,TrainData$Class)
train_missclass <- 1-sum(diag(check_train ))/sum(check_train )
#  missclassification  in traindata is approximately 2.1927621 %

train_accuracy <- 1- train_missclass
# test data accuracy is approximately  97.80724 %



#---------------------Decision Tree------------------------------------


decision_tree <- rpart(Class ~., 
                        data = TrainData,
                        method = "class")

plot(decision_tree)
text(decision_tree, all = TRUE, cex = 0.8)

prediction <- predict(decision_tree, newdata = TestData, type = "class")

#accuracy value
accuracy <- sum(prediction == TestData$Class) / nrow(TestData)

CM_decisionTree <- confusionMatrix(prediction, TestData$Class, positive = "1")

#extract the precision and recall
precision <- CM_decisionTree$byClass["Precision"]
recall <- CM_decisionTree$byClass["Recall"]

(f1_score <- 2 * (precision * recall) / (precision + recall))

#---------------------Model Comparisons------------------------------------


#plot decision tree ROC

tree_roc_score<-roc(TestData$Class, as.numeric(prediction))
plot.roc(tree_roc_score, plot=TRUE, 
         legacy.axes=TRUE, 
         percent = TRUE, xlab="False Positive Percentage", ylab="True Positive Percentage",main="ROC Curve with AUC scores for each model",
         col="#42f5dd",lwd=4,
         print.auc=TRUE,  print.auc.y=0.40)

#add glm ROC
glm_roc_score<-roc(TestData$Class,as.numeric(y_predict))
plot.roc(glm_roc_score, plot=TRUE, 
         legacy.axes=TRUE, 
         percent = TRUE, xlab="False Positive Percentage", ylab="True Positive Percentage",
         col="#ce42f5",lwd=4, add=TRUE, print.auc=TRUE, print.auc.y=0.30)

#add naive

glm_nb_score<-roc(TestData$Class,as.numeric(predicted_values))
plot.roc(glm_nb_score, plot=TRUE, 
         legacy.axes=TRUE, 
         percent = TRUE, xlab="False Positive Percentage", ylab="True Positive Percentage",
         col="#ff2e7b",lwd=4, add=TRUE,  print.auc=TRUE, print.auc.y=0.20)

#add legend
legend("topright",legend=c( "Decision Tree","Logisitic Regression", "Naive Bayes"), col=c("#42f5dd", "#ce42f5", "#ff2e7b"), lwd=1,cex = 0.5)






