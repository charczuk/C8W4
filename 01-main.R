#Practical Machine Learning Final Coursework

####################
#Load Libraries
library(lattice)
library(ggplot2)
library(caret)
library(kernlab)
library(rattle)
library(corrplot)
set.seed(1234)

##################
#Load Data
df.train <- read.csv("datasets/pml-training.csv") #load training data
df.test <- read.csv("datasets/pml-testing.csv") #load test data

###################
#Clean Data

#remove NA and metatdata
df.train <- df.train[,colMeans(is.na(df.train)) < .9]
df.train <- df.train[,-c(1:7)]

#remove columns with little variance in values
zero.var <- nearZeroVar(df.train)
df.train <- df.train[,-zero.var]

#split into a validation and training set
df.sample <- createDataPartition(y=df.train$classe, p=0.7, list=F)
train.set <- df.train[df.sample,]
validation.set <- df.train[-df.sample,]


#####################################
# Create models and get accuracy

tc <- trainControl(method="cv", number=3, verboseIter = F) #train control

# Decision Tree Model
model.decision.tree <- train(classe~., data=train.set, method="rpart", trControl = tc, tuneLength = 5)
fancyRpartPlot(model.decision.tree$finalModel)

predict.decision.tree <- predict(model.decision.tree, validation.set) #predict outcomes
confusion.matrix.decision.trees <- confusionMatrix(predict.decision.tree, factor(validation.set$classe))  #confusion matrix
confusion.matrix.decision.trees


#Random Forest Model
model.random.forest <- train(classe ~., data=train.set, method="rf", trControl = tc)

predict.random.forest <- predict(model.random.forest, validation.set)
confusion.matrix.random.forest <- confusionMatrix(predict.random.forest, factor(validation.set$classe))
confusion.matrix.random.forest


#Gradient Boosted Tree
model.gbm <- train(classe~., data = train.set, method="gbm", trControl = tc, tuneLength = 5, verbose = F)

predict.gbm <- predict(model.gbm, validation.set)
confusion.matrix.gbm <- confusionMatrix(predict.gbm, factor(validation.set$classe))
confusion.matrix.gbm


#Support Vector Machine
model.svm <- train(classe~., data= train.set, method="svmLinear", trControl = tc, tuneLength = 5, verbose = F)

predict.svm <- predict(model.svm, validation.set)
confusion.matrix.svm <- confusionMatrix(predict.svm, factor(validation.set$classe))
confusion.matrix.svm


#Accuracy
models <- c("Decision Tree Model", "Random Forest", "Gradient Boosted Tree (GBM)", "Support Vector Machine (SVM)")
accuracy <- c(round(confusion.matrix.decision.trees$overall[[1]], 3), 
              round(confusion.matrix.random.forest$overall[[1]], 3),
              round(confusion.matrix.gbm$overall[[1]], 3),
              round(confusion.matrix.svm$overall[[1]], 3))
df.accuracy <- data.frame(models, accuracy)


#Predictions on Test Set
df.predict <- predict(model.random.forest, df.test)
df.predict




