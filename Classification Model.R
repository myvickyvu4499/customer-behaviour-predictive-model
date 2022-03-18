#_______________________________________________________________________________

## PRE-PROCESSING DATA
#_______________________________________________________________________________
#install.packages("tidyverse")
#install.packages("tidymodels")
library(tidyverse)
library(tidymodels)

#import the dataset
data <- read_delim("bank.csv",  delim = ";", escape_double = FALSE, trim_ws = TRUE)
bank <- data

#remove duration column to be practical
bank <- subset(bank, select = -duration)

#remove duplicate rows -> no duplicate found
duplicate <- subset(bank, duplicated(bank))
duplicate

##treating outliers
#balance -> replace high value outliers
high <- quantile(bank$balance)[4] + 1.5*IQR(bank$balance) #get 95% percentile value
for (index in c(1:length(bank$balance))) {
  bank$balance[index] <- ifelse(bank$balance[index] > high, high, bank$balance[index])
} #replace high value outliers with the 95% percentile value for balance

#campaign -> replace high value outliers
high <- quantile(bank$campaign)[4] + 1.5*IQR(bank$campaign) #get 95% percentile value
for (index in c(1:length(bank$campaign))) {
  bank$campaign[index] <- ifelse(bank$campaign[index] > high, high, bank$campaign[index])
} #replace high value outliers with the 95% percentile value for campaign


##transforming values
#one-hot encoding
bank_ohe <- recipe(y ~ ., data = bank) %>% #start create a recipe for pre-processing data
  step_range(all_numeric()) %>% #normalise the numeric variables into same range from 0-1
  step_dummy(all_nominal_predictors()) %>% #create dummy encoding for categorical variables
  prep() #run the recipe
bank <- juice(bank_ohe) #extract the data set from recipe to bank
bank <- bank %>% relocate(y, .after = last_col()) #move y outcome column to the last

glimpse(bank) #check structure of bank after performing one-hot encoding

##assign factors for outcome variables
bank$y <- factor(bank$y)

### FEATURE SELECTION
library(caret)
library(randomForest)

# Setting control for feature selection using cross-validation random forest
control <- rfeControl(functions=rfFuncs, #using random forest
                      method="repeatedcv", #repeated cross-validation
                      number=10, #perform 10 fold
                      repeats = 3) #repeat 3 times

# Performing feature selection on bank data set after encoded
set.seed(444)
rfe <- rfe(bank[,1:41], bank$y, sizes = c(5:8,10,12,16), rfeControl=control)

# Print result -> 6 predictors found as important
print(rfe, top = 10)
plot(rfe, type=c("g", "o"), cex = 1.0) # plot the results

head(rfe$resample, 10) # show 10 resamples with 6 significant variables

# Extract 6 significant variables
predictors(rfe)
selected_vars <- predictors(rfe)

# Update the data set with only 6 significant variables
bank <- bank %>% select(all_of(selected_vars), y)

#_______________________________________________________________________________

## SAMPLING DATA
#_______________________________________________________________________________

#install packages 
#install.packages("caret")
library(caret)

#splitting into training and testing data set
set.seed(444) #reproducibility
#Create data partition with stratified sampling 
indexes <- createDataPartition(bank$y, times = 1, p = 0.7, list = FALSE)
bank.train <- bank[indexes,] 
bank.test  <- bank[-indexes,]

#Dimensions 
dim(bank.train)                
dim(bank.test)

#Check for proportion of labels in both and training and testing split
prop.table(table(bank.train$y))
prop.table(table(bank.test$y))

#_______________________________________________________________________________

## RANDOM FOREST MODEL
#_______________________________________________________________________________

# Loading library
#install.packages("caret")
#install.packages("doParallel")
library(caret)
library(doParallel)

# Setting up caret to perform 10-fold cross validation repeated 3 times
# and to use a grid search for optimal hyper parameter value (mtry)
train.control <- trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 3,
                              search = "grid",
                              allowParallel = T)

# Leverage a grid search of hyper parameter for randomForest.
tune.grid <- expand.grid(mtry = c(2:5))
#View(tune.grid)

# Running parallel
cl <- makePSOCKcluster(8)
registerDoParallel(cl)

#### TRAIN MODEL USING TRAIN SET
# Train the random forest model using 10-fold Cross validation repeated 3 times
# and a hyper parameter grid search to optimise the model
caret.cv <- train(y ~ ., 
                  data = bank.train,
                  method = "rf",
                  tuneGrid = tune.grid,
                  trControl = train.control)

stopCluster(cl)

# Print the model 
print(caret.cv)
plot(caret.cv) # plot the model performance

# Explore important variables 
print(varImp(caret.cv)) 
plot(varImp(caret.cv))

# Inserting  serial backend, to eliminate error in repetitive tasks
registerDoSEQ()

#### VALIDATE MODEL USING TEST SET
# Making  predictions using test set
preds.rf.caret <- predict(caret.cv, bank.test)

# Confusion matrix 
Confusion.rf.caret<-confusionMatrix(preds.rf.caret, bank.test$y)
Confusion.rf.caret

bank.confusion <- table(preds.rf.caret, bank.test$y)
print(bank.confusion)

## Calculate accuracy, precision, recall, F1 using formulars
# Accuracy
bank.accuracy <- sum(diag(bank.confusion)) / sum(bank.confusion)
print(bank.accuracy)

#Precision per class
bank.precision.no <- bank.confusion[1,1] / sum(bank.confusion[1,])
print(bank.precision.no)

bank.precision.yes <- bank.confusion[2,2] / sum(bank.confusion[2,])
print(bank.precision.yes)

#Overall precision
overall.precision<-(bank.precision.no+bank.precision.yes)/2
print(overall.precision)

#Recall per class
bank.recall.no <- bank.confusion[1,1] / sum(bank.confusion[,1])
print(bank.recall.no)

bank.recall.yes <- bank.confusion[2,2] / sum(bank.confusion[,2])
print(bank.recall.yes)

#Overall recall
overall.recall<-(bank.recall.no+bank.recall.yes)/2
print(overall.recall)

#F1 score
bank.f1 <- 2 * bank.precision.yes * bank.recall.yes / 
  (bank.precision.yes + bank.recall.yes)
print(bank.f1)


#_______________________________________________________________________________

## ELASTIC NET REGRESSION - LASSO AND RIDGE REGRESSION
#_______________________________________________________________________________

# Setting control parameter to train model using caret with 10-fold
# cross validation repeated 3 times and to use a grid search for optimal
# model hyper parameter values.
fitControl <- trainControl(method = "repeatedcv", 
                           number = 10, 
                           repeats = 3,
                           search = "grid",
                           savePredictions= 'final',
                           classProbs = TRUE)

# Make a custom tuning grid for Elastic Net regression model
tuneGrid <- expand.grid(alpha = seq(0, 1, length = 3), #3 alpha values ranged from 0 to 1
                        lambda = seq(0.01, 1, length = 5)) #5 lambda values ranged
#from 0.01 to 1

#### TRAIN MODEL USING TRAIN SET
# Set seed
set.seed(444)

# Train the Elastic Net regression model using 10-fold CV repeated 3 times 
# and a hyper parameter grid search to get the optimal model.
glmnet.model <- train(y~., data = bank.train, method = "glmnet",
                      tuneGrid = tuneGrid, trControl = fitControl)

print(glmnet.model)
plot(glmnet.model$finalModel, xvar = "lambda", label = TRUE)
# Best tuning parameter (hyper parameter)
glmnet.model$bestTune

# Plot the cross validation Elastic Net regression model
ggplot(glmnet.model)

# Variables importance
varImp(glmnet.model)
ggplot(varImp(glmnet.model)) #plot the variables importance

#### VALIDATE MODEL USING TEST SET
# Make predictions on the test set using the trained Elastic Net
# regression model trained on all rows of the training set
# using the found optimal hyper parameter values.
pred <- predict(glmnet.model, bank.test)

# Confusion matrix for the Elastic Net regression model
confusionMatrix(pred,bank.test$y)


## Extract the test data y to build the confusion matrix
bank.confusion <- table(pred,bank.test$y)
print(bank.confusion)

## Calculate accuracy, precision, recall, F1 using formulas
# Accuracy
bank.accuracy <- sum(diag(bank.confusion)) / sum(bank.confusion)
print(bank.accuracy)

# Precision for yes
bank.precision.yes <- bank.confusion[2,2] / sum(bank.confusion[2,])
print(bank.precision.yes)

# Recall for yes
bank.recall.yes <- bank.confusion[2,2] / sum(bank.confusion[,2])
print(bank.recall.yes)

# F1-measure
bank.f1 <- 2 * bank.precision.yes * bank.recall.yes / 
  (bank.precision.yes + bank.recall.yes)
print(bank.f1)


#_______________________________________________________________________________

## STACKING ENSEMBLE MODEL
#_______________________________________________________________________________


# Load library - caretEmsemble is used to build stacking ensemble model
# Install.packages('caretEnsemble')
library(caretEnsemble)

#### TRAIN BASE MODELS
# Setting control parameter to train base models with 10-fold cross 
# validation repeated 3 times and 30 resampling folds for every model.
ensembleControl <- trainControl(method = "repeatedcv", 
                                number = 10, 
                                repeats = 3,
                                index = createFolds(bank.train$y,30),
                                savePredictions= 'all',
                                classProbs = TRUE)

# We use random forest and elastic net regression for the base models
algorithmList <- c('rf', 'glmnet')

# Train the base models with the control parameters above
set.seed(444)
models <- caretList(y~., data=bank.train, trControl=ensembleControl, 
                    methodList=algorithmList)
results <- resamples(models)
summary(results)
dotplot(results) #plot the results of two base models 

# Check correlations between
# the base models' results
modelCor(results)
splom(results) # plot the resamples matrix of base models' results

### TRAIN META MODEL USING THE BASE MODELS' PREDICTIONS
# Setting control parameter to train model using caret with 10-fold
# cross validation repeated 3 times
stackControl <- trainControl(method = "repeatedcv", 
                             number = 10, 
                             repeats = 3,
                             savePredictions= 'final')

# Stack using logistic regression - glm family binomial
# Train with the base models' results
set.seed(444)
stack.glm <- caretStack(models, method="glm", metric="Accuracy",
                        trControl=stackControl, family = 'binomial')
summary(stack.glm)

### VALIDATE MODEL USING TEST SET
pred <- predict(stack.glm,bank.test)
# Confusion matrix for the stacking ensemble model
confusionMatrix(pred,bank.test$y)

## Extract the test data y to build the confusion matrix
bank.confusion <- table(pred,bank.test$y)
print(bank.confusion)

## Calculate accuracy, precision, recall, F1 using formulas
# Accuracy
bank.accuracy <- sum(diag(bank.confusion)) / sum(bank.confusion)
print(bank.accuracy)

# Precision for yes class

bank.precision.yes <- bank.confusion[2,2] / sum(bank.confusion[2,])
print(bank.precision.yes)

# Recall for yes class

bank.recall.yes <- bank.confusion[2,2] / sum(bank.confusion[,2])
print(bank.recall.yes)

# F1-measure
bank.f1 <- 2 * bank.precision.yes * bank.recall.yes / 
  (bank.precision.yes + bank.recall.yes)
print(bank.f1)


#________________________________________________________________________________________
#  MODEL                           ACCURACY        F1 SCORE       PRECISION      RECALL
#________________________________________________________________________________________
# Random Forest                     89.45%          25.91%          67.57%       16.03%
# Elastic Net Regression            89.60%          25.40%          72.73%       15.38%
# Stacked Generalization Ensemble   89.68%          26.32%          73.53%       16.03%
#________________________________________________________________________________________


