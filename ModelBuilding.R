library(plyr)
library(dplyr)
library(caret)
library(MASS)
library(MLmetrics)

dat = read.csv("modeldataset.csv")
names(dat)

fcols = c("PH", "IN", "MP", "Pre", "AH", "MU")
catdat = dat[, fcols]
numcols = c("Age", "Haem", "BMI")
numdat = dat[, numcols]

# Cat variables as factor
j=0
while(j < ncol(catdat)){
  j=j+1  
  catdat[,j] = as.factor(catdat[,j])
  ifelse(catdat[,j]== "1", "0", "1")
}

dat = cbind(catdat, numdat)
set.seed(123)

# Split data into 80:20 training:test ratio
train = slice_sample(.data =  dat, prop = 0.8, replace = FALSE)
prop.table(table(train$PH))
test = anti_join(dat, train)
prop.table(table(test$PH))


# Full model

# Define training control
trainCtrl = trainControl(method = "cv", number = 5)

fullmodel = train(PH ~ .,
                  data = train,
                  trControl = trainCtrl,
                  method = "glm",
                  family=binomial(), na.action = na.pass)
summary(fullmodel)
print(fullmodel)

predictfull = predict(fullmodel, data = train)
length(predictfull)
actualfull = (na.omit(train))$PH
length(actualfull)
caret::confusionMatrix(data = predictfull, reference = actualfull, mode = "everything", positive = "1")

FBeta_Score(actualfull, predictfull, beta = 2, positive = "1")

# Reduced Models (Retain p < 0.2)

# Remove multiparity. Retain BMI since its just over the threshold.
model2 = train(PH ~ IN + Pre + AH + MU + Age + Haem + BMI,
               data = train,
               trControl = trainCtrl,
               method = "glm",
               family=binomial(), na.action = na.pass)
summary(model2)
print(model2)

predict2 = predict(model2, data = train)
names(train)
length(predict2)
actualfull2 = (na.omit(train[,-c(3)]))$PH
length(actualfull2)
caret::confusionMatrix(data = predict2, reference = actualfull2, mode = "everything", positive = "1")

FBeta_Score(actualfull2, predict2, beta = 2, positive = 1)

# Removing both BMI and Multiparity
model3 = train(PH ~ IN + Pre + AH + MU + Age + Haem,
               data = train,
               trControl = trainCtrl,
               method = "glm",
               family=binomial(), na.action = na.pass)
summary(model3)
print(model3)

predict3 = predict(model3, data = train)
names(train)
length(predict3)
actualfull3 = (na.omit(train[,-c(3, 9)]))$PH
length(actualfull3)
caret::confusionMatrix(data = predict3, reference = actualfull3, mode = "everything", positive = "1")

FBeta_Score(actualfull3, predict3, beta = 2, positive = 1)

################################################################################

# Introduce new cat variables for BMI, Age and Haem

# BMI

# Introduce WHO BMI categories

# 0: below 18.5 – underweight
# 1: between 18.5 and 24.9 – normal weight
# 2: between 25 and 29.9 – overweight
# 3: between 30 and 39.9 – obese

train.cat = train
train.cat = within(train.cat, {   
  bmi.cat = NA
  bmi.cat[BMI >= 18.5 & BMI < 25] = "0" # Normal as reference
  bmi.cat[BMI < 18.5] = "1"
  bmi.cat[BMI >= 25 & BMI < 30] = "2"
  bmi.cat[BMI >= 30] = "3"
} )

names(train.cat)
fcols2 = c("PH", "IN", "MP", "Pre", "AH", "MU", "bmi.cat")
catdat2 = train.cat[, fcols2]
numcols = c("Age", "Haem", "BMI")
numdat = train.cat[, numcols]

# Cat variables as factors
j=0
while(j < ncol(catdat2)){
  j=j+1  
  catdat2[,j] = as.factor(catdat2[,j])
}

train.cat = cbind(catdat2, numdat)


model4 = train(PH ~ IN + Pre + AH + MU + Age + Haem + bmi.cat,
               data = train.cat,
               trControl = trainCtrl,
               method = "glm",
               family=binomial(), na.action = na.pass)
summary(model4)
print(model4)

predict4 = predict(model4, data = train.cat)
names(train.cat)
actualfull4 = (na.omit(train.cat[,-c(3)]))$PH
length(predict4) == length(actualfull4)
caret::confusionMatrix(data = predict4, reference = actualfull4, mode = "everything", positive = "1")

FBeta_Score(actualfull4, predict4, beta = 2, positive = 1)

# Age

train.cat = within(train.cat, {   
  age.cat = NA
  age.cat[Age < 25] = "0"
  age.cat[Age >= 25 & Age < 30] = "1"
  age.cat[Age >= 30 & Age < 35] = "2"
  age.cat[Age >= 35] = "3"
} )

model5 = train(PH ~ IN + Pre + AH + MU + age.cat + Haem + BMI,
               data = train.cat,
               trControl = trainCtrl,
               method = "glm",
               family=binomial(), na.action = na.pass)
summary(model5)
print(model5)

predict5 = predict(model5, data = train.cat)
names(train.cat)
actualfull5 = (na.omit(train.cat[,-c(7, 8)]))$PH
length(predict5) == length(actualfull5)
caret::confusionMatrix(data = predict5, reference = actualfull5, mode = "everything", positive = "1")

FBeta_Score(actualfull5, predict5, beta = 2, positive = 1)

# This improves classification compared to continuous Age

# Haem

# WHO Anemia Pregnancy classifications

# Combine into 3 levels
# <= 11g/dL (no anemia)
# 10 to < 11 g/dL (mild)
# < 10 g/dL (moderate to severe)


train.cat = within(train.cat, {   
  haem.cat = NA
  haem.cat[Haem >= 11] = "0" # No anemia
  haem.cat[Haem >= 10 & Haem < 11] = "1"
  haem.cat[Haem < 10] = "2"
} )


model6 = train(PH ~ IN + Pre + AH + MU + Age + haem.cat + BMI,
               data = train.cat,
               trControl = trainCtrl,
               method = "glm",
               family=binomial(), na.action = na.pass)
summary(model6)
print(model6)

model7 = train(PH ~ IN + Pre + AH + MU + Age + haem.cat + bmi.cat,
               data = train.cat,
               trControl = trainCtrl,
               method = "glm",
               family=binomial(), na.action = na.pass)
summary(model7)
print(model7)

model8 = train(PH ~ IN + Pre + AH + MU + age.cat + haem.cat + BMI,
               data = train.cat,
               trControl = trainCtrl,
               method = "glm",
               family=binomial(), na.action = na.pass)
summary(model8)
print(model8)

model9 = train(PH ~ IN + Pre + AH + MU + age.cat + haem.cat + bmi.cat,
               data = train.cat,
               trControl = trainCtrl,
               method = "glm",
               family=binomial(), na.action = na.pass)
summary(model9)
print(model9)

varImp(model9) # Based on the absolute value of the T statistic


predict9 = predict(model9, data = train.cat)
names(train.cat)
actualfull9 = (na.omit(train.cat[,-c(8:10)]))$PH
length(predict9) == length(actualfull9)
caret::confusionMatrix(data = predict9, reference = actualfull9, mode = "everything", positive = "1")

FBeta_Score(actualfull9, predict9, beta = 2, positive = 1)