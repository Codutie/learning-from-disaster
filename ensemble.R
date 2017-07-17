rm(list = ls())


########### PACKAGES ###########

install_and_load <- function(libraries)
{
  new.packages <- libs[!(libraries %in% installed.packages()[,"Package"])]
  if(length(new.packages)) install.packages(new.packages)
  sapply(libs, require, character.only = T, warn.conflicts = F)
}
libs <- c("plyr", "dplyr", "ggplot2", "readr", "xgboost", "caret", "Matrix", "Metrics", "miscTools", "glmnet", "caretEnsemble", "rpart")
install_and_load(libs)


########### LOAD DATA ###########

train <- read.csv("data/train.csv", stringsAsFactors = F)
test <- read.csv("data/test.csv", stringsAsFactors = F)
full <- bind_rows(train, test)
full$Survived <- ifelse(full$Survived == 1, "one", "zero") # need to do it to avoid error messages in caret trainControl classProb = T







########### STRAIGHT FORWARD FEATURE ENGINEERING ###########

# Extract Title from Names
full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)
officer <- c('Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev')
royalty <- c('Dona', 'Lady', 'the Countess','Sir', 'Jonkheer')

# Reassign mlle, ms, and mme accordingly
full$Title[full$Title == 'Mlle']        <- 'Miss' 
full$Title[full$Title == 'Ms']          <- 'Miss'
full$Title[full$Title == 'Mme']         <- 'Mrs' 
full$Title[full$Title %in% royalty]  <- 'Royalty'
full$Title[full$Title %in% officer]  <- 'Officer'

# Fare
full$Fare[is.na(full$Fare)] <- median(full[full$Pclass=='3' & full$Embarked=='S',]$Fare, na.rm=TRUE)

# Age
age_features <- c("Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked", "Title")
age_frml <- as.formula(paste("Age ~ ", paste(age_features, collapse= "+")))
age_fit <- rpart(age_frml, data = full[-which(is.na(full$Age)), ], cp = 0.001)
full$Age[is.na(full$Age)] <- round(predict(age_fit, full[is.na(full$Age), ]), 2)

# Family Size
full$FSize <- full$SibSp + full$Parch + 1

# Child
full$Child[full$Age < 18] <- 'Child'
full$Child[full$Age >= 18] <- 'Adult'

# FsizeD
full$FSizeD[full$FSize == 1] <- 'Alone'
full$FSizeD[full$FSize < 5 & full$FSize > 1] <- 'Small'
full$FSizeD[full$FSize > 4] <- 'Big'

# Embarked
full$Embarked[c(62, 830)] <- 'C'



########### ENSEMBLE MODELING ###########

features <- c("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title", "FSizeD", "Child")
frml <- as.formula(paste("Survived ~ ", paste(features, collapse= "+")))


# prepare data for modeling
char_features <- names(full)[sapply(full, is.character)]
full[, char_features] <- lapply(full[, char_features], as.factor)
train <- full[!is.na(full$Survived), ]
test <- full[is.na(full$Survived), ]; test$Survived <- NULL


my_control <- trainControl(
  method="boot",
  repeats=5,
  number=25,
  verboseIter = F,
  savePredictions=TRUE,
  summaryFunction=twoClassSummary,
  classProbs = T,
  index=createResample(train$Survived, 25)
)

set.seed(121)
model_list <- caretList(
  frml, 
  data=train,
  metric="ROC",
  trControl=my_control,
  tuneList = list(
     rf = caretModelSpec(method = "rf", tuneGrid = data.frame(mtry = round(sqrt(length(features))))),
     nnet = caretModelSpec(method = "nnet"),
     adaboost = caretModelSpec(method = "adaboost")
     ))


modelCor(resamples(model_list))

#           rf      nnet  adaboost
# rf       1.0000000 0.3447546 0.7782943
# nnet     0.3447546 1.0000000 0.2417687
# adaboost 0.7782943 0.2417687 1.0000000

model_preds <- lapply(model_list, predict, newdata=test, type="prob")
model_preds <- lapply(model_preds, function(x) x[, 1])
model_preds <- data.frame(model_preds)


# glm ensemble

set.seed(1)
ensemble <- caretStack(
  model_list,
  method="glm",
  metric="ROC",
  trControl=trainControl(
    method="boot",
    number=20,
    savePredictions="final",
    classProbs=TRUE,
    summaryFunction=twoClassSummary
  )
)

ens_preds <- predict(ensemble, newdata=test, type="prob")
model_preds$ensemble <- ens_preds
# caTools::colAUC(model_preds, testing$Survived)



########### PREDICTION ###########

preds <- predict(ensemble, newdata = test, type = "raw")
preds <- ifelse(preds == "zero", 0, 1)


# submission
submission <- data.frame(PassengerId = test$PassengerId, Survived = preds)
write.csv(submission, "submission.csv", row.names = F)









# set.seed(5)
# inTrain <- createDataPartition(y = train$Survived, p = .8, list = FALSE)
# training <- train[ inTrain,]
# testing <- train[-inTrain,]