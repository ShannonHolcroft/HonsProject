# PACKAGES #

library(xgboost)
library(dplyr)
library(readr)
library(stringr)
library(caret)
library(car)
library(devtools)
library(Matrix)
library(data.table)
library(ggpubr)
library(tidyr)
library(pROC)
library(ROCR)
library(plotROC)
library(glmnet)

# Set seed

# seed = 2022
seed = 1993

# Read in data

dataSet = read.csv("modelsubset.csv")
dat = dataSet
dat = dat[, -1]

factor.cols = c("PH", "IN", "MP", "Pre", "AH", "MU", "IFD", "BMI", "age.cat", "haem.cat")
factordat = dat[, factor.cols]
numdat = dat[, c("Age", "Haem")] # Removing continuous BMI here


# Cat variables as factors
j=0
while(j < ncol(factordat)){
  j=j+1  
  factordat[,j] = as.factor(factordat[,j])
}

dat = cbind(factordat, numdat)

# Split data into 70:20:10 training:val:test ratio
set.seed(seed)
trainI = createDataPartition(dat$PH, p = 0.7, list = FALSE)

trainSet = dat[trainI,]
prop.table(table(trainSet$PH))

remainder = dat[-trainI, ]
set.seed(seed)
valI = createDataPartition(remainder$PH, p = 0.2*nrow(dat)/nrow(remainder), list = FALSE)
valSet = remainder[valI, ]

prop.table(table(valSet$PH))


testSet = remainder[-valI, ]
prop.table(table(testSet$PH))

# Proportions of cases are similar to original data set

# Model feature sets
set1 = c('PH', 'IN', 'MP', 'Pre','AH','MU', 'IFD', 'BMI','Age', 'Haem')
set2 = c('PH', 'IN', 'MP', 'Pre','AH','MU', 'IFD', 'BMI','age.cat', 'Haem')
set3 = c('PH', 'IN', 'MP', 'Pre','AH','MU', 'IFD', 'BMI', 'Age', 'haem.cat')
set4 = c('PH', 'IN', 'MP', 'Pre','AH','MU','IFD', 'BMI','age.cat', 'haem.cat')

# Model training sets
train1 = na.omit(trainSet[, c(which(colnames(trainSet) %in% set1))])
train2 = na.omit(trainSet[, c(which(colnames(trainSet) %in% set2))])
train3 = na.omit(trainSet[, c(which(colnames(trainSet) %in% set3))])
train4 = na.omit(trainSet[, c(which(colnames(trainSet) %in% set4))])

# Training response vectors
y1 = ifelse(as.numeric(train1$PH) == 1, "Absent", "Present")
y2 = ifelse(as.numeric(train2$PH) == 1, "Absent", "Present")
y3 = ifelse(as.numeric(train3$PH) == 1, "Absent", "Present")
y4 = ifelse(as.numeric(train4$PH) == 1, "Absent", "Present")

# Model validation sets
val1 = na.omit(valSet[, c(which(colnames(valSet) %in% set1))])
val2 = na.omit(valSet[, c(which(colnames(valSet) %in% set2))])
val3 = na.omit(valSet[, c(which(colnames(valSet) %in% set3))])
val4 = na.omit(valSet[, c(which(colnames(valSet) %in% set4))])

# Validation response vectors
yval1 = ifelse(as.numeric(val1$PH) == 1, "Absent", "Present")
yval2 = ifelse(as.numeric(val2$PH) == 1, "Absent", "Present")
yval3 = ifelse(as.numeric(val3$PH) == 1, "Absent", "Present")
yval4 = ifelse(as.numeric(val3$PH) == 1, "Absent", "Present")

# Model test sets
test1 = na.omit(testSet[, c(which(colnames(testSet) %in% set1))])
test2 = na.omit(testSet[, c(which(colnames(testSet) %in% set2))])
test3 = na.omit(testSet[, c(which(colnames(testSet) %in% set3))])
test4 = na.omit(testSet[, c(which(colnames(testSet) %in% set4))])

# Model training

  # Define training control

  # Specify cross-validation method and number of folds

# Logistic regression, elastic net, RF

trainCtrl =  trainControl(method = "cv",
                          number = 5,
                          classProbs = TRUE,
                          savePredictions = "final",
                          summaryFunction = multiClassSummary)

# Two-class summary doesn't give Kappa

# XGBoost

xgb_trcontrol1 = trainControl(
                method = "cv",
                number = 5,
              classProbs = TRUE,
              summaryFunction = mnLogLoss,
              allowParallel = TRUE,
              verboseIter = FALSE,
                returnData = FALSE)

# Logistic Regression #

# Try out each basic model

# Model 1

logm1 = glm(PH ~ ., train1, family = "binomial")
summary(logm1)

# Remove BMI
logm1 = glm(PH ~ IN + MP + Pre + AH + MU + IFD + Age + Haem, train1, family = "binomial")
summary(logm1)

# Remove Age
logm1 = glm(PH ~ IN + MP + Pre + AH + MU + IFD + Haem, train1, family = "binomial")
summary(logm1)

# Training performance
m1_pred = predict(logm1, train1, type = "response")
m1_class = as.factor(ifelse(m1_pred < 0.5, 0, 1))
confmat1 = confusionMatrix(m1_class, train1$PH, positive = "1")
confmat1

k1 = confmat1$overall[['Kappa']]
se1 = confmat1$byClass[['Sensitivity']]
sp1 = confmat1$byClass[['Specificity']]

# Model 2
logm2 = glm(PH ~ ., train2, family = "binomial")
summary(logm2)

# Remove Age
logm2 = glm(PH ~ IN + MP + Pre + AH + MU + IFD + BMI + Haem, train2, family = "binomial")
summary(logm2)

# Remove BMI
logm2 = glm(PH ~ IN + MP + Pre + AH + MU + IFD + Haem, train2, family = "binomial")
summary(logm2)

# Produces the same model
k2 = k1
se2 = se1
sp2 = sp1

# Model 3
logm3 = glm(PH ~ ., train3, family = "binomial")
summary(logm3)

logm3 = glm(PH ~ IN + MP + Pre + AH + MU + IFD + Age + haem.cat, train3, family = "binomial")
summary(logm3)

# Training performance
m3_pred = predict(logm3, train3, type = "response")
m3_class = as.factor(ifelse(m3_pred < 0.5, 0, 1))
confmat3 = confusionMatrix(m3_class, train3$PH, positive = "1")
confmat3

k3 = confmat3$overall[['Kappa']]
se3 = confmat3$byClass[['Sensitivity']]
sp3 = confmat3$byClass[['Specificity']]


# Model 4
logm4 = glm(PH ~ ., train4, family = "binomial")
summary(logm4)

# Remove haemoglobin level
logm4 = glm(PH ~ IN + MP + Pre + AH + MU + IFD + age.cat + BMI, train4, family = "binomial")

# Remove BMI
logm4 = glm(PH ~ IN + MP + Pre + AH + MU + IFD + age.cat, train4, family = "binomial")
summary(logm4)

# Training performance
m4_pred = predict(logm4, train4, type = "response")
m4_class = as.factor(ifelse(m4_pred < 0.5, 0, 1))
confmat4 = confusionMatrix(m4_class, train4$PH, positive = "1")
confmat4

k4 = confmat4$overall[['Kappa']]
se4 = confmat4$byClass[['Sensitivity']]
sp4 = confmat4$byClass[['Specificity']]

max(k1, k2, k3, k4)
max(se1, se2, se3, se4)
max(sp1, sp2, sp3, sp4)

# Train best basic model after feature selection

# Train Model 1

set.seed(seed)
train1$PH = y1
logmod = train(PH ~ IN + MP + Pre + AH + MU + IFD + Haem,
               data = train1,
               trControl = trainCtrl,
               method = "glm", metric = "Kappa",
               family=binomial)

logresults = logmod$results

# Training performance
train1$PH = as.factor(y1)
m1_pred = predict(logmod, train1, type = "prob")
m1_class = as.factor(ifelse(m1_pred[, 2] < 0.5, "Absent", "Present"))
trainconfmat = confusionMatrix(m1_class, train1$PH, positive = "Present")
trainconfmat

# Validation performance
val1$PH = as.factor(yval1)
m1_valpred = predict(logmod, val1, type = "prob")
m1_valclass = as.factor(ifelse(m1_valpred[, 2] < 0.5, "Absent", "Present"))
valconfmat = confusionMatrix(m1_valclass, val1$PH, positive = "Absent")
valconfmat

# Performs worse in val with this default threshold

# Threshold selection

# Get optimal training threshold using ROC
train1$PH = as.factor(y1)
roc1best = plot(roc(train1$PH, m1_pred[,2], print.thresh = TRUE))
t1.1 = coords(roc1best, x = "best", transpose = TRUE)

# Plot ROC curve

# Create ROC dataframe for plotting
logrocdf = data.frame(cbind(train1$PH, m1_pred[,2]))
names(logrocdf) = c("response", "prob")
logrocdf$response = ifelse(logrocdf$response == 1, "Absent", "Present")

# Plot ROC curve using ggplot and plotROC
logrocplot = ggplot(logrocdf, aes(d = response, m = prob)) + geom_roc(labels = FALSE) + 
                    style_roc(theme = theme_bw, xlab = "\n 1 - Specificity \n", ylab = "\n Sensitivity \n") +
                    geom_point(x = 1 - t1.1[[2]], y = t1.1[[3]], col = "black") +
                    geom_vline(xintercept = 1 - t1.1[[2]], col = "black", lty = 2) + 
                    geom_text(x = 0.25, y = 0.9, label = "* 0.288", colour = "black")

# Validation of ROC threshold

# Validation performance
m1_valpred = predict(logmod, val1, type = "prob")
m1_valclass = as.factor(ifelse(m1_valpred[, 2] < t1.1[[1]], "Absent", "Present"))
valconfmat = confusionMatrix(m1_valclass, val1$PH, positive = "Present")

# ROC training threshold performs well in validation

# Try out a range between ROC threshold and default of 0.5
threshvec1 = seq(from = t1.1[[1]], to = 0.5, by = 0.025)

# Training and validation predicted probabilities 

log_probs = predict(logmod, train1, type = 'prob')
log_valprobs = predict(logmod, val1, type = 'prob')

# Store performance for each threshold in training and validation

kaptrain = c()
kapval = c()
senstrain = c()
sensval = c()
spectrain = c()
specval = c()

for (i in threshvec1) {
  
  # Training
  
  log_class = as.factor(ifelse(log_probs[,2] < i, "Absent", "Present"))
  confmat1tuned = confusionMatrix(log_class,train1$PH, positive = "Present")
  
  kaptrain = c(kaptrain, confmat1tuned$overall[['Kappa']])
  senstrain = c(senstrain, confmat1tuned$byClass[['Sensitivity']])
  spectrain = c(spectrain, confmat1tuned$byClass[['Specificity']])
  
  # Validation

  log_valclass = as.factor(ifelse(log_valprobs[,2] < i, "Absent", "Present"))
  valtunedconf1 = confusionMatrix(log_valclass, val1$PH, positive = "Present")
  
  kapval = c(kapval, valtunedconf1$overall[['Kappa']])
  sensval = c(sensval, valtunedconf1$byClass[['Sensitivity']])
  specval = c(specval, valtunedconf1$byClass[['Specificity']])
}

# Store performance results from different thresholds
trainres1 = cbind(kaptrain, kapval, senstrain, sensval, spectrain, specval)

# Create long data frames for plotting response curves
trainres1 = data.frame(cbind(threshvec1, trainres1))
names(trainres1) = c("Threshold", "TrainKappa", "ValKappa", "TrainSens", "ValSens", "TrainSpec", "ValSpec")


log_threshval_long = data.frame(pivot_longer(trainres1, TrainKappa:ValSpec))
log_kappa_long = log_threshval_long[log_threshval_long$name == "TrainKappa"| log_threshval_long$name == "ValKappa", ]
log_senspec_long = anti_join(log_threshval_long, log_kappa_long)

# Kappa tuning plot

logkappatune = ggplot(log_kappa_long, aes(x = Threshold, y = value)) +
                      theme_bw() + geom_smooth(aes(col = name), se = FALSE) +
                      scale_colour_discrete(name = "", labels = c("Kappa (Train)", "Kappa (Val)")) +
                      geom_vline(xintercept = threshvec1[4], lty = 2) + xlim(0.25, 0.5) +
                      labs(x = "\n Threshold \n", y = "\n Kappa Statistic \n")

# Sensitivity & specificity tuning plot

logsenspectune =  ggplot(log_senspec_long, aes(x = Threshold, y = value)) +
                        theme_bw() + geom_smooth(aes(col = name), se = FALSE) + 
                        scale_colour_discrete(name = "", labels = c("Sensitivity (Train)", "Specificity (Train)", "Sensitivity (Val)", "Specificity (Val)")) +
                        geom_vline(xintercept = threshvec1[4], lty = 2) + xlim(0.25, 0.5) +
                        labs(x = "\n Threshold \n", y = "\n Sensitivity/Specificity \n", legend = "")

ggarrange(logkappatune, logsenspectune, nrow = 2, ncol = 1)

# Store predicted probabilities for model averaging

# Training
modav1_trainprobs = predict(logmod, train1, type = 'prob')[2]
tunedclass3 = as.factor(ifelse(modav1_trainprobs< threshvec1[4], "Absent", "Present"))
confusionMatrix(tunedclass3, train1$PH, positive = "Present")

# Validation
modav2_valprobs = predict(logmod, val1, type = 'prob')[2]
tunedclass4 = as.factor(ifelse(modav2_valprobs< threshvec1[4], "Absent", "Present"))
confusionMatrix(tunedclass4, val1$PH, positive = "Present")

#####################################################################################################################################################################

# ELASTIC NET #

# Note that Glmnet performs standardization unless otheriwse specified
# Glmnet returns coefficients on original scale

# Create a hyperparameter tuning grid for alpha and lambda
tunegrid1 = expand.grid(alpha = seq(0, 1, length = 6),
                        lambda = seq(0, 1, length = 6))

# Model 1
train1$PH = as.factor(y1)
set.seed(seed)
full1_enet = train(
  PH ~ ., data = train1,
  method = "glmnet", metric = "Kappa",
  trControl = trainCtrl, tuneGrid = tunegrid1)

full1results = full1_enet$results

best1 = full1results[full1results$alpha == full1_enet$bestTune$alpha & 
               full1results$lambda == full1_enet$bestTune$lambda, ]

# Model 2
train2$PH = as.factor(y2)
full2_enet = train(
  PH ~ ., data = train2,
  method = "glmnet", metric = "Kappa",
  trControl = trainCtrl, tuneGrid = tunegrid1
)

full2results = full2_enet$results

best2 = full2results[full2results$alpha == full2_enet$bestTune$alpha & 
                       full2results$lambda == full2_enet$bestTune$lambda, ]

ggplot(full2_enet) + theme_bw()

# Model 3
train3$PH = as.factor(y3)
set.seed(seed)
full3_enet = train(
  PH ~ ., data = train3,
  method = "glmnet", metric = "Kappa",
  trControl = trainCtrl, tuneGrid = tunegrid1
)

full3results = full3_enet$results

best3 = full3results[full3results$alpha == full3_enet$bestTune$alpha & 
                       full3results$lambda == full3_enet$bestTune$lambda, ]

# Model 4
train4$PH = as.factor(y4)
set.seed(seed)
full4_enet = train(
  PH ~ ., data = train4,
  method = "glmnet", metric = "Kappa",
  trControl = trainCtrl, tuneGrid = tunegrid1
)

full4results = full4_enet$results
best4 = full4results[full4results$alpha == full4_enet$bestTune$alpha & 
                       full4results$lambda == full4_enet$bestTune$lambda, ]

elasticbest = rbind(best1, best2, best3, best4)

# Threshold selection for the best model

# Training and validation predicted probabilities 

p2 = predict(full2_enet, train2, type = "prob")
p2_class = as.factor(ifelse(p2[, 2] < 0.5, "Absent", "Present"))
trainconfmat = confusionMatrix(p2_class, train2$PH, positive = "Present")
trainconfmat

vp2 = predict(full2_enet, val2, type = "prob")
vp2_class = as.factor(ifelse(vp2[, 2] < 0.5, "Absent", "Present"))
valconfmat = confusionMatrix(vp2_class, as.factor(yval2), positive = "Present")
valconfmat

# Threshold selection

# Get optimal training threshold using ROC curve
roc1best = plot(roc(as.factor(y2), p2[, 2], print.thresh = TRUE))
t1.1 = coords(roc1best, x = "best", transpose = TRUE)

# Create ROC dataframe
regrocdf = data.frame(cbind(y2, p2))
names(regrocdf) = c("response", "prob")

# Plot ROC curve using ggplot and plotRC
regrocplot = ggplot(regrocdf, aes(d = response, m = prob)) + geom_roc(labels = FALSE) + 
                    style_roc(theme = theme_grey, xlab = "1 - Specificity", ylab = "Sensitivity") +
                    geom_point(x = 1 - t1.1[[2]], y = t1.1[[3]], col = "black") +
                    geom_vline(xintercept = 1 - t1.1[[2]], col = "black", lty = 2) +
                    geom_text(x = 0.15, y = 0.9, label = "* 0.372", colour = "black")

# Validation of threshold

# Validation performance
p2_valpred = predict(full2_enet, val2, type = "prob")[, 2]
p2_valclass = as.factor(ifelse(p2_valpred < t1.1[[1]], "Absent", "Present"))
valconfmat = confusionMatrix(p2_valclass, as.factor(yval2), positive = "Present")
valconfmat

# Try out a range between ROC threshold and default of 0.5
threshvec2 = seq(from = t1.1[[1]], to = 0.5, by = 0.025)

# Store performance for each threshold in training and validation

kaptrain = c()
kapval = c()
senstrain = c()
sensval = c()
spectrain = c()
specval = c()

for (i in threshvec2) {
  
  # Training
  
  reg_class = as.factor(ifelse(p2[, 2] < i, "Absent", "Present"))
  confmat1tuned = confusionMatrix(reg_class, as.factor(y2), positive = "Present")
  
  kaptrain = c(kaptrain, confmat1tuned$overall[['Kappa']])
  senstrain = c(senstrain, confmat1tuned$byClass[['Sensitivity']])
  spectrain = c(spectrain, confmat1tuned$byClass[['Specificity']])
  
  # Validation
  
  reg_valclass = as.factor(ifelse(vp2[, 2] < i, "Absent", "Present"))
  valtunedconf1 = confusionMatrix(reg_valclass, as.factor(yval2), positive = "Present")
  kapval = c(kapval, valtunedconf1$overall[['Kappa']])
  sensval = c(sensval, valtunedconf1$byClass[['Sensitivity']])
  specval = c(specval, valtunedconf1$byClass[['Specificity']])
  
}

trainres2 = cbind(kaptrain, kapval, senstrain, sensval, spectrain, specval)
trainres2

# Create long data frames
trainres2 = data.frame(cbind(threshvec2, trainres2))
names(trainres2) = c("Threshold", "TrainKappa", "ValKappa", "TrainSens", "ValSens", "TrainSpec", "ValSpec")

reg_threshval_long = data.frame(pivot_longer(trainres2, TrainKappa:ValSpec))
reg_kappa_long = reg_threshval_long[reg_threshval_long$name == "TrainKappa"| reg_threshval_long$name == "ValKappa", ]
reg_senspec_long = anti_join(reg_threshval_long, reg_kappa_long)

regkappatune = ggplot(reg_kappa_long, aes(x = Threshold, y = value)) +
                      theme_bw() + geom_smooth(aes(col = name), se = FALSE) +
                      scale_colour_discrete(name = "", labels = c("Kappa (Train)", "Kappa (Val)")) +
                      geom_vline(xintercept = threshvec2[2], lty = 2) + xlim(0.32, 0.5) +
                      labs(x = "\n Threshold \n", y = "\n Kappa Statistic \n")

regsenspectune =  ggplot(reg_senspec_long, aes(x = Threshold, y = value)) +
                        theme_bw() + geom_smooth(aes(col = name), se = FALSE) + 
                        scale_colour_discrete(name = "", labels = c("Sensitivity (Train)", "Specificity (Train)", "Sensitivity (Val)", "Specificity (Val)")) +
                        geom_vline(xintercept = threshvec2[2], lty = 2) + xlim(0.32, 0.5) +
                        labs(x = "\n Threshold \n", y = "\n Sensitivity/Specificity \n", legend = "")

ggarrange(regkappatune, regsenspectune, nrow = 2, ncol = 1)

# Predicted probabilities for model averaging

# Training performance
modav2_trainprobs = predict(full2_enet, train2, type = 'prob')

# Validation performance
modav2_valprobs = predict(full2_enet, val2, type = 'prob')

##################################################################################################################################################################

# RANDOM FOREST #

# Tuning

# Random Forest

rfGrid = expand.grid(mtry = 2:8, splitrule = "gini", min.node.size = c(10, 15, 20, 25))

# Model 1

train1$PH = as.factor(y1)

# Max depth 10
set.seed(seed)
rf11 = train(PH~.,
             data = train1,
             method = "ranger",
             importance= "impurity",
             metric = "Kappa",
             num.trees = 1000,
             max.depth = 10,
             tuneGrid = rfGrid,
             trControl = trainCtrl)


rf11results = rf11$results

bestrf11 = rf11results[rf11results$mtry == rf11$bestTune$mtry & 
                        rf11results$splitrule == rf11$bestTune$splitrule &
                        rf11results$min.node.size == rf11$bestTune$min.node.size, ]

# Max depth 6
set.seed(seed)
rf12 = train(PH~.,
             data = train1,
             method = "ranger",
             importance= "impurity",
             metric = "Kappa",
             num.trees = 1000,
             max.depth = 6,
             tuneGrid = rfGrid,
             trControl = trainCtrl)

rf12results = rf12$results

bestrf12 = rf12results[rf12results$mtry == rf12$bestTune$mtry & 
                        rf12results$splitrule == rf12$bestTune$splitrule &
                        rf12results$min.node.size == rf12$bestTune$min.node.size, ]

# Max depth 2
set.seed(seed)
rf13 = train(PH~.,
             data = train1,
             method = "ranger",
             importance= "impurity",
             metric = "Kappa",
             num.trees = 1000,
             max.depth = 2,
             tuneGrid = rfGrid,
             trControl = trainCtrl)

rf13results = rf13$results

bestrf13 = rf13results[rf13results$mtry == rf13$bestTune$mtry & 
                         rf13results$splitrule == rf13$bestTune$splitrule &
                         rf13results$min.node.size == rf13$bestTune$min.node.size, ]

# Model 2

train2$PH = as.factor(y2)

# Max depth 10
set.seed(seed)
rf21 = train(PH~.,
             data = train2,
             method = "ranger",
             importance= "impurity",
             metric = "Kappa",
             num.trees = 1000,
             max.depth = 10,
             tuneGrid = rfGrid,
             trControl = trainCtrl)

rf21results = rf21$results

bestrf21 = rf21results[rf21results$mtry == rf21$bestTune$mtry & 
                         rf21results$splitrule == rf21$bestTune$splitrule &
                         rf21results$min.node.size == rf21$bestTune$min.node.size, ]

# Max depth 6
set.seed(seed)
rf22 = train(PH~.,
             data = train2,
             method = "ranger",
             importance= "impurity",
             metric = "Kappa",
             num.trees = 1000,
             max.depth = 6,
             tuneGrid = rfGrid,
             trControl = trainCtrl)

rf22results = rf22$results

bestrf22 = rf22results[rf22results$mtry == rf22$bestTune$mtry & 
                         rf22results$splitrule == rf22$bestTune$splitrule &
                         rf22results$min.node.size == rf22$bestTune$min.node.size, ]
# Max depth 2
set.seed(seed)
rf23 = train(PH~.,
             data = train2,
             method = "ranger",
             importance= "impurity",
             metric = "Kappa",
             num.trees = 1000,
             max.depth = 2,
             tuneGrid = rfGrid,
             trControl = trainCtrl)

rf23results = rf23$results

bestrf23 = rf23results[rf23results$mtry == rf23$bestTune$mtry & 
                         rf23results$splitrule == rf23$bestTune$splitrule &
                         rf23results$min.node.size == rf23$bestTune$min.node.size, ]

# Model 3
train3$PH = as.factor(y3)


# Max depth 10
set.seed(seed)
rf31 = train(PH~.,
             data = train3,
             method = "ranger",
             importance= "impurity",
             metric = "Kappa",
             num.trees = 1000,
             max.depth = 10,
             tuneGrid = rfGrid,
             trControl = trainCtrl)

rf31results = rf31$results

bestrf31 = rf31results[rf31results$mtry == rf31$bestTune$mtry & 
                         rf31results$splitrule == rf31$bestTune$splitrule &
                         rf31results$min.node.size == rf31$bestTune$min.node.size, ]

# Max depth 6
set.seed(seed)
rf32 = train(PH~.,
             data = train3,
             method = "ranger",
             importance= "impurity",
             metric = "Kappa",
             num.trees = 1000,
             max.depth = 6,
             tuneGrid = rfGrid,
             trControl = trainCtrl)

rf32results = rf32$results

bestrf32 = rf32results[rf32results$mtry == rf32$bestTune$mtry & 
                         rf32results$splitrule == rf32$bestTune$splitrule &
                         rf32results$min.node.size == rf32$bestTune$min.node.size, ]

# Max depth 2
set.seed(seed)
rf33 = train(PH~.,
             data = train3,
             method = "ranger",
             importance= "impurity",
             metric = "Kappa",
             num.trees = 1000,
             max.depth = 2,
             tuneGrid = rfGrid,
             trControl = trainCtrl)

rf33results = rf33$results

bestrf33 = rf33results[rf33results$mtry == rf33$bestTune$mtry & 
                         rf33results$splitrule == rf33$bestTune$splitrule &
                         rf33results$min.node.size == rf33$bestTune$min.node.size, ]

# Model 4

train4$PH = as.factor(y4)

# Max depth 10
set.seed(seed)
rf41 = train(PH~.,
             data = train4,
             method = "ranger",
             importance= "impurity",
             metric = "Kappa",
             num.trees = 1000,
             max.depth = 10,
             tuneGrid = rfGrid,
             trControl = trainCtrl)

rf41results = rf41$results

bestrf41 = rf41results[rf41results$mtry == rf41$bestTune$mtry & 
                         rf41results$splitrule == rf41$bestTune$splitrule &
                         rf41results$min.node.size == rf41$bestTune$min.node.size, ]


# Max depth 6
set.seed(seed)
rf42 = train(PH~.,
             data = train4,
             method = "ranger",
             importance= "impurity",
             metric = "Kappa",
             num.trees = 1000,
             max.depth = 6,
             tuneGrid = rfGrid,
             trControl = trainCtrl)

rf42results = rf42$results

bestrf42 = rf42results[rf42results$mtry == rf42$bestTune$mtry & 
                         rf42results$splitrule == rf42$bestTune$splitrule &
                         rf42results$min.node.size == rf42$bestTune$min.node.size, ]

# Max depth 2
rf43 = train(PH~.,
             data = train4,
             method = "ranger",
             importance= "impurity",
             metric = "Kappa",
             num.trees = 1000,
             max.depth = 2,
             tuneGrid = rfGrid,
             trControl = trainCtrl)

rf43results = rf43$results

bestrf43 = rf43results[rf43results$mtry == rf43$bestTune$mtry & 
                         rf43results$splitrule == rf43$bestTune$splitrule &
                         rf43results$min.node.size == rf43$bestTune$min.node.size, ]

bestrf = rbind(bestrf11, bestrf12, bestrf13, bestrf21, bestrf22, bestrf23, bestrf31, bestrf32,
               bestrf33, bestrf41, bestrf42, bestrf43)

ggplot(rf12) + theme_bw()

# Training performance and probabilities
rf1_pred = predict(rf12, train1)
rf1_probs = predict(rf12, train1, type = 'prob')[ , 2]
rf1_class = as.factor(ifelse(rf1_probs < 0.5, "Absent", "Present"))
confmatrf = confusionMatrix(rf1_class, train1$PH, positive = "Present")
confmatrf

# Validation performance and probabilities
rf1_valpred = predict(rf12, train1)
rf1_valprobs = predict(rf12, val1, type = 'prob')[ , 2]
rf1_valclass = as.factor(ifelse(rf1_valprobs < 0.5, "Absent", "Present"))
confmatrf = confusionMatrix(rf1_class, train1$PH, positive = "Present")
confmatrf

# Threshold selection

# Get optimal training threshold using ROC curves
roc1best = plot(roc(train1$PH, rf1_probs, print.thresh = TRUE))
t1.1 = coords(roc1best, x = "best", transpose = TRUE)
# Create ROC dataframe
rfrocdf = data.frame(cbind(train1$PH, rf1_probs))
names(rfrocdf) = c("response", "prob")


# Plot ROC curve using ggplot and plotRC
rfrocplot = ggplot(rfrocdf, aes(d = response, m = prob)) + geom_roc(labels = FALSE) + 
  style_roc(theme = theme_bw, xlab = "\n 1 - Specificity \n", ylab = "\n Sensitivity \n") +
  geom_point(x = 1 - t1.1[[2]], y = t1.1[[3]], col = "black") +
  geom_vline(xintercept = 1 - t1.1[[2]], col = "black", lty = 2) + 
  geom_text(x = 0.23, y = 0.95, label = "* 0.243", colour = "black")

# Try out a range between ROC threshold and default of 0.5
threshvec3 = seq(from = t1.1[[1]], to = 0.5, by = 0.025)

# Store performance for each threshold in training and validation

kaptrain = c()
kapval = c()
senstrain = c()
sensval = c()
spectrain = c()
specval = c()

for (i in threshvec3) {
  # Training
  rf_class = as.factor(ifelse(rf1_probs < i, "Absent", "Present"))
  confmat1tuned = confusionMatrix(rf_class, train1$PH, positive = "Present")
  kaptrain = c(kaptrain, confmat1tuned$overall[['Kappa']])
  senstrain = c(senstrain, confmat1tuned$byClass[['Sensitivity']])
  spectrain = c(spectrain, confmat1tuned$byClass[['Specificity']])

  # Validation
  rf_valclass = as.factor(ifelse(rf1_valprobs < i, "Absent", "Present"))
  valtunedconf1 = confusionMatrix(rf_valclass, val1$PH, positive = "Present")
  kapval = c(kapval, valtunedconf1$overall[['Kappa']])
  sensval = c(sensval, valtunedconf1$byClass[['Sensitivity']])
  specval = c(specval, valtunedconf1$byClass[['Specificity']])

}

trainres3 = data.frame(cbind(threshvec3, kaptrain, kapval, senstrain, sensval, spectrain, specval))
names(trainres3) = c("Threshold", "TrainKappa", "ValKappa", "TrainSens", "ValSens", "TrainSpec", "ValSpec")


# Training probs for Model Averaging
rf_trainprobs_ave = predict(rf12, train1, type = 'prob')[, 2]
write.csv(rf_trainprobs_ave, "rftrain.csv", row.names = FALSE)

# Val probs for Model Averaging
rf_valprobs_ave = predict(rf11, val1, type = 'prob')[, 2]
write.csv(rf_valprobs_ave, "rfval.csv", row.names = FALSE)

# Create long data frames

rf_threshval_long = data.frame(pivot_longer(trainres3, TrainKappa:ValSpec))
rf_kappa_long = rf_threshval_long[rf_threshval_long$name == "TrainKappa"| rf_threshval_long$name == "ValKappa", ]
rf_senspec_long = anti_join(rf_threshval_long, rf_kappa_long)

rfkappatune =  ggplot(rf_kappa_long, aes(x = Threshold, y = value)) +
  theme_bw() + geom_smooth(aes(col = name), se = FALSE) +
  scale_colour_discrete(name = "", labels = c("Kappa (Train)", "Kappa (Val)")) +
  xlim(0.20, 0.52) + geom_vline(xintercept = threshvec3[3], lty = 2) +
  labs(x = "\n Threshold \n", y = "\n Kappa Statistic \n")

rfsenspectune =  ggplot(rf_senspec_long, aes(x = Threshold, y = value)) +
  theme_bw() + geom_smooth(aes(col = name), se = FALSE)  + xlim(0.20, 0.52) +
  scale_colour_discrete(name = "", labels = c("Sensitivity (Train)", "Specificity (Train)", "Sensitivity (Val)", "Specificity (Val)")) +
  labs(x = "\n Threshold \n", y = "\n Sensitivity/Specificity \n", legend = "") +
  geom_vline(xintercept = threshvec3[3], lty = 2)

ggarrange(rfkappatune, rfsenspectune, nrow = 2, ncol = 1)

###########################################################################################################################################

# ExtraTrees

etGrid = expand.grid(mtry = 2:8, splitrule = "extratrees", min.node.size = c(10, 15, 20, 25))

# Model 1

# Max depth 10
train1$PH = as.factor(y1)
set.seed(seed)
et11 = train(PH~.,
             data = train1,
             method = "ranger",
             importance= "impurity",
             metric = "Kappa",
             num.trees = 1000,
             max.depth = 10,
             replace = FALSE,
             sample.fraction = 1,
             tuneGrid = etGrid,
             trControl = trainCtrl)

et11results = et11$results

bestet11 = et11results[et11results$mtry == et11$bestTune$mtry & 
                         et11results$splitrule == et11$bestTune$splitrule &
                         et11results$min.node.size == et11$bestTune$min.node.size, ]


# Max depth 6
set.seed(seed)
et12 = train(PH~.,
             data = train1,
             method = "ranger",
             importance= "impurity",
             metric = "Kappa",
             num.trees = 1000,
             max.depth = 6,
             replace = FALSE,
             sample.fraction = 1,
             tuneGrid = etGrid,
             trControl = trainCtrl)

et12results = et12$results

bestet12 = et12results[et12results$mtry == et12$bestTune$mtry & 
                         et12results$splitrule == et12$bestTune$splitrule &
                         et12results$min.node.size == et12$bestTune$min.node.size, ]

# Max depth 2
set.seed(seed)
et13 = train(PH~.,
             data = train1,
             method = "ranger",
             importance= "impurity",
             metric = "Kappa",
             num.trees = 1000,
             max.depth = 2,
             replace = FALSE,
             sample.fraction = 1,
             tuneGrid = etGrid,
             trControl = trainCtrl)

et13results = et13$results

bestet13 = et13results[et13results$mtry == et13$bestTune$mtry & 
                         et13results$splitrule == et13$bestTune$splitrule &
                         et13results$min.node.size == et13$bestTune$min.node.size, ]

# Model 2
train2$PH = as.factor(y2)

# Max depth 10
set.seed(seed)
et21 = train(PH~.,
             data = train2,
             method = "ranger",
             importance= "impurity",
             metric = "Kappa",
             num.trees = 1000,
             max.depth = 10,
             replace = FALSE,
             sample.fraction = 1,
             tuneGrid = etGrid,
             trControl = trainCtrl)

et21results = et21$results

bestet21 = et21results[et21results$mtry == et21$bestTune$mtry & 
                         et21results$splitrule == et21$bestTune$splitrule &
                         et21results$min.node.size == et21$bestTune$min.node.size, ]


# Max depth 6
set.seed(seed)
et22 = train(PH~.,
             data = train2,
             method = "ranger",
             importance= "impurity",
             metric = "Kappa",
             num.trees = 1000,
             max.depth = 6,
             replace = FALSE,
             sample.fraction = 1,
             tuneGrid = etGrid,
             trControl = trainCtrl)

et22results = et22$results

bestet22 = et22results[et22results$mtry == et22$bestTune$mtry & 
                         et22results$splitrule == et22$bestTune$splitrule &
                         et22results$min.node.size == et22$bestTune$min.node.size, ]


# Max depth 2
et23 = train(PH~.,
             data = train2,
             method = "ranger",
             importance= "impurity",
             metric = "Kappa",
             num.trees = 1000,
             max.depth = 2,
             replace = FALSE,
             sample.fraction = 1,
             tuneGrid = etGrid,
             trControl = trainCtrl)

et23results = et23$results

bestet23 = et23results[et23results$mtry == et23$bestTune$mtry & 
                         et23results$splitrule == et23$bestTune$splitrule &
                         et23results$min.node.size == et23$bestTune$min.node.size, ]

# Model 3

train3$PH = as.factor(y3)

# Max depth 10
set.seed(seed)
et31 = train(PH~.,
             data = train3,
             method = "ranger",
             importance= "impurity",
             metric = "Kappa",
             num.trees = 1000,
             max.depth = 10,
             replace = FALSE,
             sample.fraction = 1,
             tuneGrid = etGrid,
             trControl = trainCtrl)

et31results = et31$results

bestet31 = et31results[et31results$mtry == et31$bestTune$mtry & 
                         et31results$splitrule == et31$bestTune$splitrule &
                         et31results$min.node.size == et31$bestTune$min.node.size, ]

# Max depth 6
set.seed(seed)
et32 = train(PH~.,
             data = train3,
             method = "ranger",
             importance= "impurity",
             metric = "Kappa",
             num.trees = 1000,
             max.depth = 6,
             replace = FALSE,
             sample.fraction = 1,
             tuneGrid = etGrid,
             trControl = trainCtrl)

et32results = et32$results

bestet32 = et32results[et32results$mtry == et32$bestTune$mtry & 
                         et32results$splitrule == et32$bestTune$splitrule &
                         et32results$min.node.size == et32$bestTune$min.node.size, ]
# Max depth 2
set.seed(seed)
et33 = train(PH~.,
             data = train3,
             method = "ranger",
             importance= "impurity",
             metric = "Kappa",
             num.trees = 1000,
             max.depth = 2,
             replace = FALSE,
             sample.fraction = 1,
             tuneGrid = etGrid,
             trControl = trainCtrl)

et33results = et33$results

bestet33 = et33results[et33results$mtry == et33$bestTune$mtry & 
                         et33results$splitrule == et33$bestTune$splitrule &
                         et33results$min.node.size == et33$bestTune$min.node.size, ]

# Model 4
train4$PH = as.factor(y4)

# Max depth 10
set.seed(seed)
et41 = train(PH~.,
             data = train4,
             method = "ranger",
             importance= "impurity",
             metric = "Kappa",
             num.trees = 1000,
             max.depth = 10,
             replace = FALSE,
             sample.fraction = 1,
             tuneGrid = etGrid,
             trControl = trainCtrl)

et41results = et41$results

bestet41 = et41results[et41results$mtry == et41$bestTune$mtry & 
                         et41results$splitrule == et41$bestTune$splitrule &
                         et41results$min.node.size == et41$bestTune$min.node.size, ]
# Max depth 6
set.seed(seed)
et42 = train(PH~.,
             data = train4,
             method = "ranger",
             importance= "impurity",
             metric = "Kappa",
             num.trees = 1000,
             max.depth = 6,
             replace = FALSE,
             sample.fraction = 1,
             tuneGrid = etGrid,
             trControl = trainCtrl)

et42results = et42$results
bestet42 = et42results[et42results$mtry == et42$bestTune$mtry & 
                         et42results$splitrule == et42$bestTune$splitrule &
                         et42results$min.node.size == et42$bestTune$min.node.size, ]

# Max depth 2
set.seed(seed)
et43 = train(PH~.,
             data = train4,
             method = "ranger",
             importance= "impurity",
             metric = "Kappa",
             num.trees = 1000,
             max.depth = 2,
             replace = FALSE,
             sample.fraction = 1,
             tuneGrid = etGrid,
             trControl = trainCtrl)

et43results = et43$results

bestet43 = et41results[et43results$mtry == et43$bestTune$mtry & 
                         et43results$splitrule == et43$bestTune$splitrule &
                         et43results$min.node.size == et43$bestTune$min.node.size, ]

bestet = rbind(bestet11, bestet12, bestet13, bestet21, bestet22, bestet23, bestet31, bestet32,
               bestet33, bestet41, bestet42, bestet43)

ggplot(et11) + theme_bw()

# Training and validation predicted probabilities

et1_pred = predict(et11, train1)
et1_probs = predict(et11, train1, type = 'prob')[, 2]
confmater = confusionMatrix(et1_pred, train1$PH, positive = "Present")
confmater

et1_valprobs = predict(et11, val1, type = 'prob')[, 2]

# Threshold selection

# Get optimal training threshold using ROC curve
roc1best = plot(roc(train1$PH, et1_probs, print.thresh = TRUE))
t1.1 = coords(roc1best, x = "best", transpose = TRUE)

# Create ROC dataframe
ertrocdf = data.frame(cbind(y1, et1_probs))
names(ertrocdf) = c("response", "prob")

# Plot ROC curve using ggplot and plotRC
ertrocplot = ggplot(rfrocdf, aes(d = response, m = prob)) + geom_roc(labels = FALSE) + 
  style_roc(theme = theme_bw, xlab = "\n 1 - Specificity \n", ylab = "\n Sensitivity \n") +
  geom_point(x = 1 - t1.1[[2]], y = t1.1[[3]], col = "black") +
  geom_vline(xintercept = 1 - t1.1[[2]], col = "black", lty = 2) + 
  geom_text(x = 0.35, y = 0.75, label = "* 0.197", colour = "black")

# Try out a range between ROC threshold and default of 0.5
threshvec4 = seq(from = t1.1[[1]], to = 0.5, by = 0.025)

# Store performance for each threshold in training and validation

kaptrain = c()
kapval = c()
senstrain = c()
sensval = c()
spectrain = c()
specval = c()

for (i in threshvec4) {
  
  # Training
  er_class = as.factor(ifelse(et1_probs < i, "Absent", "Present"))
  confmat1tuned = confusionMatrix(er_class, train1$PH, positive = "Present")
  kaptrain = c(kaptrain, confmat1tuned$overall[['Kappa']])
  senstrain = c(senstrain, confmat1tuned$byClass[['Sensitivity']])
  spectrain = c(spectrain, confmat1tuned$byClass[['Specificity']])

  # Validation
  er_valclass = as.factor(ifelse(et1_valprobs < i, "Absent", "Present"))
  valtunedconf1 = confusionMatrix(er_valclass, val1$PH, positive = "Present")
  kapval = c(kapval, valtunedconf1$overall[['Kappa']])
  sensval = c(sensval, valtunedconf1$byClass[['Sensitivity']])
  specval = c(specval, valtunedconf1$byClass[['Specificity']])
  
}

trainres4 = data.frame(cbind(threshvec4, kaptrain, kapval, senstrain, sensval, spectrain, specval))
names(trainres4) = c("Threshold", "TrainKappa", "ValKappa", "TrainSens", "ValSens", "TrainSpec", "ValSpec")

# Training probs for Model Averaging
er_trainprobs_ave = predict(et11, train1, type = 'prob')[, 2]
write.csv(er_trainprobs_ave, "ertrain.csv", row.names = FALSE)

# Val probs for Model Averaging
er_valprobs_ave = predict(et11, val1, type = 'prob')[, 2]
write.csv(er_valprobs_ave, "erval.csv", row.names = FALSE)

# Create long data frames for plotting
et_threshval_long = data.frame(pivot_longer(trainres4, TrainKappa:ValSpec))
et_kappa_long = et_threshval_long[et_threshval_long$name == "TrainKappa"| et_threshval_long$name == "ValKappa", ]
et_senspec_long = anti_join(et_threshval_long, et_kappa_long)

etkappatune =  ggplot(et_kappa_long, aes(x = Threshold, y = value)) +
                      theme_bw() + geom_smooth(aes(col = name), se = FALSE) +
                      scale_colour_discrete(name = "", labels = c("Kappa (Train)", "Kappa (Val)")) +
                      xlim(0.20, 0.52) + geom_vline(xintercept = threshvec4[4], lty = 2) +
                      labs(x = "\n Threshold \n", y = "\n Kappa Statistic \n")

etsenspectune =  ggplot(et_senspec_long, aes(x = Threshold, y = value)) +
                        theme_bw() + geom_smooth(aes(col = name), se = FALSE)  + xlim(0.20, 0.52) +
                        scale_colour_discrete(name = "", labels = c("Sensitivity (Train)", "Specificity (Train)", "Sensitivity (Val)", "Specificity (Val)")) +
                        labs(x = "\n Threshold \n", y = "\n Sensitivity/Specificity \n", legend = "") +
                        geom_vline(xintercept = threshvec4[4], lty = 2)

ggarrange(etkappatune, etsenspectune, nrow = 2, ncol = 1)

###############################################################################################################################################################

# XGBOOST #

# Make training data numeric

X_train1 = model.matrix(PH ~ ., data = train1)[,-1]
X_train2 = model.matrix(PH ~ ., data = train2)[,-1]
X_train3 = model.matrix(PH ~ ., data = train3)[,-1]
X_train4 = model.matrix(PH ~ ., data = train4)[,-1]


# Response vectors
y1 = ifelse(train1$PH == "Absent", 0, 1)
y2 = ifelse(train2$PH == "Absent", 0, 1)
y3 = ifelse(train3$PH == "Absent", 0, 1)
y4 = ifelse(train4$PH == "Absent", 0, 1)

# Grid space to search
xgbGrid = expand.grid(nrounds = 2000, 
                      max_depth = c(2, 4, 6),
                      eta = c(0.01, 0.1, 0.3),
                      gamma = c(0.01, 0.1, 0.3, 0.5),
                      colsample_bytree = c(0.5, 0.8, 1),
                      min_child_weight = c(1, 2, 4),
                      subsample = c(0.5, 0.8, 1))

# Try out each basic model before tuning

# Model 1
xgb1 = xgboost(data = X_train1, 
               label = y1,
               nrounds = 1000,
               objective = "binary:logistic")


# Training performance
xgb1_probs = predict(xgb1, X_train1)
xgb1_pred = ifelse(xgb1_probs < 0.5, 0, 1)
xgb1_pred = recode_factor(xgb1_pred, `0` = "0", `1` = "1")
y1 = recode_factor(y1, `0` = "0", `1` = "1")

confmat1 = confusionMatrix(xgb1_pred, y1, positive = "1")
confmat1

k1 = confmat1$overall[['Kappa']]
se1 = confmat1$byClass[['Sensitivity']]
sp1 = confmat1$byClass[['Specificity']]

# Model 2

xgb2 = xgboost(data = X_train2, 
               label = as.numeric(y2),
               nrounds = 1000,
               objective = "binary:logistic")


# Training performance
xgb2_probs = predict(xgb2, X_train2)
xgb2_pred = ifelse(xgb2_probs < 0.5, 0, 1)
xgb2_pred = recode_factor(xgb2_pred, `0` = "0", `1` = "1")
y2 = recode_factor(y2, `0` = "0", `1` = "1")

confmat2 = confusionMatrix(xgb2_pred, y2, positive = "1")
confmat2

k2 = confmat2$overall[['Kappa']]
se2 = confmat2$byClass[['Sensitivity']]
sp2 = confmat2$byClass[['Specificity']]

# Model 3

xgb3 = xgboost(data = X_train3, 
               label = y3,
               nrounds = 1000,
               objective = "binary:logistic")

# Training performance
xgb3_probs = predict(xgb3, X_train3)
xgb3_pred = ifelse(xgb3_probs < 0.5, 0, 1)
xgb3_pred = recode_factor(xgb3_pred, `0` = "0", `1` = "1")
y3 = recode_factor(y3, `0` = "0", `1` = "1")

confmat3 = confusionMatrix(xgb3_pred, y3, positive = "1")
confmat3

k3 = confmat3$overall[['Kappa']]
se3 = confmat3$byClass[['Sensitivity']]
sp3 = confmat3$byClass[['Specificity']]

# Model 4

xgb4 = xgboost(data = X_train4, 
               label = y4,
               nrounds = 1000,
               objective = "binary:logistic")


# Training performance
xgb4_probs = predict(xgb4, X_train4)
xgb4_pred = ifelse(xgb4_probs < 0.5, 0, 1)
xgb4_pred = recode_factor(xgb4_pred, `0` = "0", `1` = "1")
y4 = recode_factor(y4, `0` = "0", `1` = "1")

confmat4 = confusionMatrix(xgb4_pred, y4, positive = "1")
confmat4

k4 = confmat4$overall[['Kappa']]
se4 = confmat4$byClass[['Sensitivity']]
sp4 = confmat4$byClass[['Specificity']]


max(k1, k2, k3, k4)
max(se1, se2, se3, se4)
max(sp1, sp2, sp3, sp4)

# Tune hyperparameters in Model 3

# Response vectors
y3 = train3$PH

set.seed(seed)
xgbfit3 = train(PH ~ ., data = train3,
                method = 'xgbTree', metric = 'logLoss',
                trControl = xgb_trcontrol1,
                verbose = FALSE,
                tuneGrid = xgbGrid)
#load("xgbfit3gamma.Rdata")

xgb3_res = xgbfit3$results
xgb3_bt = xgbfit3$bestTune

# Training and validation predicted probabilities and classes

# Training
xgb3_probs = predict(xgbfit3, train3, type = "prob")[, 2]
xgb3_pred = as.factor(ifelse(xgb3_probs < 0.5, "Absent", "Present"))

confmat3 = confusionMatrix(xgb3_pred, as.factor(y3), positive = "Present")
confmat3

# Validation
xgb3_valprobs = predict(xgbfit3, val3, type = "prob")[, 2]
xgb3_valpred = as.factor(ifelse(xgb3_valprobs < 0.5, "Absent", "Present"))

confmat3val = confusionMatrix(xgb3_valpred, as.factor(yval3), positive = "Present")
confmat3val

# Threshold selection

# Get optimal training threshold using ROC
roc3best = roc(y3, xgb3_probs, print.thresh = TRUE)
t3.1 = coords(roc3best, x = "best")

# Create ROC dataframe
xgrocdf = data.frame(cbind(y3, xgb3_probs))
names(xgrocdf) = c("response", "prob")

# Plot ROC curve using ggplot and plotRC
xgrocplot = ggplot(xgrocdf, aes(d = response, m = prob)) + geom_roc(labels = FALSE) + 
                    style_roc(theme = theme_bw, xlab = "1 - Specificity", ylab = "Sensitivity") +
                    geom_point(x = 1 - t3.1[[2]], y = t3.1[[3]], col = "black") +
                    geom_vline(xintercept = 1 - t3.1[[2]], col = "black", lty = 4) + 
                    geom_text(x = 0.3, y = 0.95, label = "* 0.256", colour = "black")

# Identify classification threshold

# Try out a range between ROC threshold and default of 0.5
threshvec5 = seq(from = t3.1[[1]], to = 0.5, by = 0.025)

# Store performance for each threshold in training and validation

kaptrain = c()
kapval = c()
senstrain = c()
sensval = c()
spectrain = c()
specval = c()

for (i in threshvec5) {
  
  # Training
  xb_class = as.factor(ifelse(xgb3_probs < i, "Absent", "Present"))
  confmat1tuned = confusionMatrix(xb_class, train3$PH, positive = "Present")
  kaptrain = c(kaptrain, confmat1tuned$overall[['Kappa']])
  senstrain = c(senstrain, confmat1tuned$byClass[['Sensitivity']])
  spectrain = c(spectrain, confmat1tuned$byClass[['Specificity']])

  # Validation
  xb_valclass = as.factor(ifelse(xgb3_valprobs < i, "Absent", "Present"))
  valtunedconf1 = confusionMatrix(xb_valclass, as.factor(yval3), positive = "Present")
  kapval = c(kapval, valtunedconf1$overall[['Kappa']])
  sensval = c(sensval, valtunedconf1$byClass[['Sensitivity']])
  specval = c(specval, valtunedconf1$byClass[['Specificity']])
}

trainres5 = data.frame(cbind(threshvec5, kaptrain, kapval, senstrain, sensval, spectrain, specval))
names(trainres5) = c("Threshold", "TrainKappa", "ValKappa", "TrainSens", "ValSens", "TrainSpec", "ValSpec")

# Create long data frames
xgb_threshval_long = data.frame(pivot_longer(trainres5, TrainKappa:ValSpec))
xgb_kappa_long = xgb_threshval_long[xgb_threshval_long$name == "TrainKappa"| xgb_threshval_long$name == "ValKappa", ]
xgb_senspec_long = anti_join(xgb_threshval_long, xgb_kappa_long)


xgbkappatune =  ggplot(xgb_kappa_long, aes(x = Threshold, y = value)) +
                      theme_bw() + geom_smooth(aes(col = name), se = FALSE) +
                      scale_colour_discrete(name = "", labels = c("Kappa (Train)", "Kappa (Val)")) +
                      xlim(0.20, 0.52) + geom_vline(xintercept = threshvec5[4], lty = 2) +
                      labs(x = "\n Threshold \n", y = "\n Kappa Statistic \n")


xgb_senspectune =  ggplot(xgb_senspec_long, aes(x = Threshold, y = value)) +
                    theme_bw() + geom_smooth(aes(col = name), se = FALSE) + xlim(0.20, 0.52) +
                    scale_colour_discrete(name = "", labels = c("Sensitivity (Train)", "Specificity (Train)", "Sensitivity (Val)", "Specificity (Val)")) +
                    labs(x = "\n Threshold \n", y = "\n Sensitivity/Specificity \n", legend = "") +
                    geom_vline(xintercept = threshvec5[4], lty = 2)

ggarrange(xgbkappatune, xgb_senspectune, nrow = 2, ncol = 1)

##############################################################################################################

# MODEL AVERAGING #

MA_train = read_xlsx("MA_train.xlsx")
MA_val = read_xlsx("MA_val.xlsx")

probstrain = matrix(as.numeric(c(MA_train$Logistic, MA_train$ElasticNet, MA_train$RF, MA_train$ERT, MA_train$XGBoost)), ncol = 5)
probstrain = data.frame(probstrain)
names(probstrain) = c("Logit", "Elastic", "RF", "ERT", "XGB")

probsval = matrix(as.numeric(c(MA_val$Logit, MA_val$ElasticNet, MA_val$RF, MA_val$ERT, MA_val$XGBoost)), ncol = 5)
probsval = data.frame(probsval)
names(probsval) = c("Logit", "Elastic", "RF", "ERT", "XGB")

# Average predictions

# Training
scaled = (1/5)*probstrain
scaled$combo = apply(scaled, 1, sum)
aveprobs = scaled$combo
aveprobs = aveprobs[-c(284:292)]

# Validation
scaledval = (1/5)*probsval
scaledval$combo = apply(scaledval, 1, sum)
avevalprobs = scaledval$combo

# Basic prediction in training
avepred = as.factor(ifelse(aveprobs < 0.5, "Absent", "Present"))
confusionMatrix(avepred, train1$PH, positive = "Present")

# Basic prediction in validation
avevalpred = as.factor(ifelse(avevalprobs < 0.5, "Absent", "Present"))
confusionMatrix(avevalpred, val1$PH, positive = "Present")

# Better validation performance

# Threshold selection

# Get optimal training threshold using ROC
roc1best = plot(roc(train1$PH, aveprobs, print.thresh = TRUE))
t1.1 = coords(roc1best, x = "best", transpose = TRUE)

# Create ROC dataframe
averocdf = data.frame(cbind(y1, aveprobs))
names(averocdf) = c("response", "prob")

# Plot ROC curve using ggplot and plotRC
averocplot = ggplot(averocdf, aes(d = response, m = prob)) + geom_roc(labels = FALSE) + 
                    style_roc(theme = theme_bw, xlab = "\n 1 - Specificity \n", ylab = "\n Sensitivity \n") +
                    geom_point(x = 1 - t1.1[[2]], y = t1.1[[3]], col = "black") +
                    geom_vline(xintercept = 1 - t1.1[[2]], col = "black", lty = 2) + 
                    geom_text(x = 0.2, y = 0.9, label = "* 0.348", colour = "black")


# Validation performance with new threshold
avevalpred = as.factor(ifelse(avevalprobs < t1.1[[1]], "Absent", "Present"))
valtunedconf1 = confusionMatrix(avevalpred, val1$PH, positive = "Present")
valtunedconf1


# Try out a range between ROC threshold and default of 0.5
threshvec6 = seq(from = t1.1[[1]], to = 0.5, by = 0.025)

# Store performance for each threshold in training and validation

kaptrain = c()
kapval = c()
senstrain = c()
sensval = c()
spectrain = c()
specval = c()

for (i in threshvec6) {
  
  # Training
  aveclass = as.factor(ifelse(aveprobs < i, "Absent", "Present"))
  confmat1tuned = confusionMatrix(aveclass, train1$PH, positive = "Present")
  kaptrain = c(kaptrain, confmat1tuned$overall[['Kappa']])
  senstrain = c(senstrain, confmat1tuned$byClass[['Sensitivity']])
  spectrain = c(spectrain, confmat1tuned$byClass[['Specificity']])

  # Validation
  avevalclass = as.factor(ifelse(avevalprobs < i, "Absent", "Present"))
  valtunedconf1 = confusionMatrix(avevalclass, val1$PH, positive = "Present")
  kapval = c(kapval, valtunedconf1$overall[['Kappa']])
  sensval = c(sensval, valtunedconf1$byClass[['Sensitivity']])
  specval = c(specval, valtunedconf1$byClass[['Specificity']])

}

trainres6 = cbind(kaptrain, kapval, senstrain, sensval, spectrain, specval)

# Create long data frames

trainres6 = data.frame(cbind(threshvec6, trainres6))
names(trainres6) = c("Threshold", "TrainKappa", "ValKappa", "TrainSens", "ValSens", "TrainSpec", "ValSpec")

log_threshval_long = data.frame(pivot_longer(trainres6, TrainKappa:ValSpec))
log_kappa_long = log_threshval_long[log_threshval_long$name == "TrainKappa"| log_threshval_long$name == "ValKappa", ]
log_senspec_long = anti_join(log_threshval_long, log_kappa_long)


avekappatune = ggplot(log_kappa_long, aes(x = Threshold, y = value)) +
              theme_bw() + geom_smooth(aes(col = name), se = FALSE) +
              scale_colour_discrete(name = "", labels = c("Kappa (Train)", "Kappa (Val)")) +
              geom_vline(xintercept = threshvec6[1], lty = 2) + xlim(0.34, 0.5) +
              labs(x = "\n Threshold \n", y = "\n Kappa Statistic \n")

avesenspectune =  ggplot(log_senspec_long, aes(x = Threshold, y = value)) +
                  theme_bw() + geom_smooth(aes(col = name), se = FALSE) + 
                  scale_colour_discrete(name = "", labels = c("Sensitivity (Train)", "Specificity (Train)", "Sensitivity (Val)", "Specificity (Val)")) +
                  geom_vline(xintercept = threshvec6[1], lty = 2) + xlim(0.34, 0.5) +
                  labs(x = "\n Threshold \n", y = "\n Sensitivity/Specificity \n", legend = "")

ggarrange(avekappatune, avesenspectune, nrow = 2)

##########################################################################################################################################################

# Comparison of Model Performance

# Misclassifications

# Logistic Regression

t1 = 0.343

# Training performance
train1$PH = as.factor(y1)
m1_class = as.factor(ifelse(m1_pred[, 2] < t1, "Absent", "Present"))
trainconfmat = confusionMatrix(m1_class, train1$PH, positive = "Present")

# Validation performance
val1$PH = as.factor(yval1)
m1_valclass = as.factor(ifelse(m1_valpred[, 2] < t1, "Absent", "Present"))
valconfmat = confusionMatrix(m1_valclass, val1$PH, positive = "Present")

logitmc_t = (35+25)/283
logitmc_v = (6+7)/83

logit = trainres1[4, -1]
logitmiss = c(logitmc_t, logitmc_v)
names(logitmiss) = c("TrainMisclass", "ValMisClass")
logit = data.frame(c(logitmiss, logit))

# Elastic Net

t2 = 0.394

# Training performance
train2$PH = as.factor(y2)
m2_class = as.factor(ifelse(p2[, 2] < t2, "Absent", "Present"))
trainconfmat = confusionMatrix(m2_class, train2$PH, positive = "Present")

# Validation performance
val2$PH = as.factor(yval2)
m2_valclass = as.factor(ifelse(p2_valpred < t2, "Absent", "Present"))
valconfmat = confusionMatrix(m2_valclass, val2$PH, positive = "Absent")

enetmc_t = (38+14)/283
enetmc_v = (8+3)/83

enet = trainres2[2, -1]
enetmiss = c(enetmc_t, enetmc_v)
names(enetmiss) = c("TrainMisclass", "ValMisClass")
enet = data.frame(c(enetmiss, enet))

# Random Forest

t3 = 0.293

# Training performance
train1$PH = as.factor(y1)
m3_class = as.factor(ifelse(rf1_probs < t3, "Absent", "Present"))
trainconfmat = confusionMatrix(m3_class, train1$PH, positive = "Present")

# Validation performance
val1$PH = as.factor(yval1)
m3_valclass = as.factor(ifelse(rf1_valprobs < t3, "Absent", "Present"))
valconfmat = confusionMatrix(m3_valclass, val1$PH, positive = "Present")

rfmc_t = (7+34)/283
rfmc_v = (6+20)/83

rf = trainres3[3, -1]
rfmiss = c(rfmc_t, rfmc_v)
names(rfmiss) = c("TrainMisclass", "ValMisClass")
rf = data.frame(c(rfmiss, rf))

# Extremely Randomised Trees

t4 = 0.272

# Training performance
train1$PH = as.factor(y1)
m4_class = as.factor(ifelse(et1_probs< t4, "Absent", "Present"))
trainconfmat = confusionMatrix(m4_class, train1$PH, positive = "Present")

# Validation performance
val1$PH = as.factor(yval1)
m4_valclass = as.factor(ifelse(et1_valprobs < t4, "Absent", "Present"))
valconfmat = confusionMatrix(m4_valclass, val1$PH, positive = "Present")


ermc_t = (27+30)/283
ermc_v = (6+11)/83

(5+7)/83
er = trainres4[4, -1]
ermiss = c(ermc_t, ermc_v)
names(ermiss) = c("TrainMisclass", "ValMisClass")
er = data.frame(c(ermiss, er))

# XGBoost

t5 = 0.331

# Training performance
train3$PH = as.factor(y3)
m5_class = as.factor(ifelse(xgb3_probs < t5, "Absent", "Present"))
trainconfmat = confusionMatrix(m5_class, train3$PH, positive = "Present")

# Validation performance
val3$PH = as.factor(yval3)
m5_valclass = as.factor(ifelse(xgb3_valprobs < t5, "Absent", "Present"))
valconfmat = confusionMatrix(m5_valclass, val3$PH, positive = "Present")

xgmc_t = (36+21)/283
xgmc_v = (9+4)/83

xg = trainres5[4, -1]
xgmiss = c(xgmc_t, xgmc_v)
names(xgmiss) = c("TrainMisclass", "ValMisClass")
xg = data.frame(c(xgmiss, xg))

# Model averaging

t6 = 0.348


# Training performance
train1$PH = as.factor(y1)
m6_class = as.factor(ifelse(aveprobs < t6, "Absent", "Present"))
trainconfmat = confusionMatrix(m6_class, train1$PH, positive = "Present")

# Validation performance
val1$PH = as.factor(yval1)
m6_valclass = as.factor(ifelse(avevalprobs< t6, "Absent", "Present"))
valconfmat = confusionMatrix(m6_valclass, val1$PH, positive = "Present")

avemc_t = (31+14)/283
avemc_v = (14+5)/83

ave = trainres6[1, -1]
avemiss = c(avemc_t, avemc_v)
names(avemiss) = c("TrainMisclass", "ValMisClass")
ave = data.frame(c(avemiss, ave))

allres = rbind(logit, enet, rf, er, xg, ave)
nam = c("Logistic Regression", "Elastic Net", "Random Forest", "Extremely Randomised Trees", "XGBoost","Model Averaging")
allres = cbind(nam, allres)

trainres = allres[, c(1, 2, 4, 6, 8)]
names(trainres) = c("Names", "Misclassification", "Kappa", "Sensitivity", "Specificity")
valres = allres[, c(1, 3, 5, 7, 9), ]
names(valres) = c("Names", "Misclassification", "Kappa", "Sensitivity", "Specificity")

# Training

misclasses = ggplot(trainres, aes(x = Names, y = Misclassification)) +
  geom_bar(stat = "identity") +
  geom_col(width = 0.1)+
  coord_flip() + scale_y_continuous(name="") +
  theme(axis.text.x = element_text(face="bold", color= "black",
                                   size=8, angle=0),
        axis.text.y = element_text(face="bold", color= "black",
                                   size=8, angle=0)) +
  labs(x = "", title = "\n Misclass. Rate \n")


kap = ggplot(trainres, aes(x = Names, y = Kappa)) +
  geom_bar(stat = "identity") +
  geom_col(width = 0.1)+
  coord_flip() + scale_y_continuous(name="") +
  theme(axis.text.x = element_text(face="bold", color= "black",
                                   size=8, angle=0),
        axis.text.y = element_text(face="bold", color= "black",
                                   size=8, angle=0)) +
  labs(x = "", title = "\n Kappa Statistic \n")


sens = ggplot(trainres, aes(x = Names, y = Sensitivity)) +
  geom_bar(stat = "identity") +
  geom_col(width = 0.1)+
  coord_flip() + scale_y_continuous(name="") +
  theme(axis.text.x = element_text(face="bold", color= "black",
                                   size=8, angle=0),
        axis.text.y = element_text(face="bold", color= "black",
                                   size=8, angle=0)) + labs(x = "", title = "\n Sensitivity \n")


spec = ggplot(trainres, aes(x = Names, y = Specificity)) +
  geom_bar(stat = "identity") +
  geom_col(width = 0.1)+
  coord_flip() + scale_y_continuous(name="") +
  theme(axis.text.x = element_text(face="bold", color= "black",
                                   size=8, angle=0),
        axis.text.y = element_text(face="bold", color= "black",
                                   size=8, angle=0)) +
  labs(x = "", y = "\n Specificity \n", title = "\n Specificity \n")

ggarrange(misclasses, kap, spec, sens, nrow = 2, ncol = 2)

misclasses_val = ggplot(valres, aes(x = Names, y = Misclassification)) +
  geom_bar(stat = "identity") +
  geom_col(width = 0.1)+
  coord_flip() + scale_y_continuous(name="") +
  theme(axis.text.x = element_text(face="bold", color= "black",
                                   size=8, angle=0),
        axis.text.y = element_text(face="bold", color= "black",
                                   size=8, angle=0)) +
  labs(x = "", title = "\n Misclass. Rate \n")
misclasses_val

kap_val = ggplot(valres, aes(x = Names, y = Kappa)) +
  geom_bar(stat = "identity") +
  geom_col(width = 0.1)+
  coord_flip() + scale_y_continuous(name="") +
  theme(axis.text.x = element_text(face="bold", color= "black",
                                   size=8, angle=0),
        axis.text.y = element_text(face="bold", color= "black",
                                   size=8, angle=0)) +
  labs(x = "", title = "\n Kappa Statistic \n")

kap_val

sens_val = ggplot(valres, aes(x = Names, y = Sensitivity)) +
  geom_bar(stat = "identity") +
  geom_col(width = 0.1)+
  coord_flip() + scale_y_continuous(name="") +
  theme(axis.text.x = element_text(face="bold", color= "black",
                                   size=8, angle=0),
        axis.text.y = element_text(face="bold", color= "black",
                                   size=8, angle=0)) + labs(x = "", title = "\n Sensitivity \n")

spec_val = ggplot(valres, aes(x = Names, y = Specificity)) +
  geom_bar(stat = "identity") +
  geom_col(width = 0.1)+
  coord_flip() + scale_y_continuous(name="") +
  theme(axis.text.x = element_text(face="bold", color= "black",
                                   size=8, angle=0),
        axis.text.y = element_text(face="bold", color= "black",
                                   size=8, angle=0)) +
  labs(x = "", title = "\n Specificity \n")

ggarrange(misclasses_val, kap_val,spec_val,sens_val, nrow = 2, ncol = 2)

################################################################################################################################################################

# Testing the best performing elastic net model

test2 = na.omit(testSet[, c(which(colnames(testSet) %in% set2))])
X_test2 = model.matrix(PH ~ ., data = test2)[,-1]
ytest2 = as.factor(ifelse(as.numeric(test2$PH) == 1, "Absent", "Present"))
test2$PH = ytest2

# Applying it to final test model
Final_test = glmnet(X_test2, ytest2, family = "binomial", alpha = 0.2 , lambda = 0, standardize = TRUE) # optimal lambda and alpha used

testpredict = predict(Final_test, X_test2, type = "response")
ptest2_class = as.factor(ifelse(testpredict < 0.397 , "Absent", "Present")) # 0.397 is validated optimal threshold
testconfmat = confusionMatrix(ptest2_class, ytest2, positive = "Present")
testconfmat

# Visual description of results :
df = data.frame(Frequency=c(0.081,0.749, 0.750, 0.966),
                 Stats_nam=c("Misclass. Rate", "Kappa", "Sensitivity", "Specificity"))


ggplot(data=df, aes(x=Stats_nam, y=Frequency)) +
  geom_bar(stat="identity", fill="darkgrey", width = 0.7)+ theme_bw() +
  geom_text(aes(label=Frequency), vjust=1.6, color="black", size=3.5)+
  theme(axis.text.x = element_text(face="bold", color= "black",
                                   size=8, angle=0),
        axis.text.y = element_text(face="bold", color= "black",
                                   size=8, angle=0))+ xlab("") + ylab("")

# Variable importance taken from trained model

# Define training control with no performance metric

trainCtrl2 = trainControl(method = "none",
                          summaryFunction = multiClassSummary,
                          classProbs = TRUE, savePredictions = "final")

# Define tune grid with only optimal hyperparameter values

tunegrid2 = expand.grid(alpha = 0.2,
                        lambda = 0)

test_enet = train(
  PH ~ ., data = test2,
  method = "glmnet",
  trControl = trainCtrl2, tuneGrid = tunegrid2
)

varimpdat = varImp(test_enet)$importance
varimpdat$Names = c("Med Insurance", "Multiparity",
                    "Previous PPH",
                    "Antepartum Haemorrhage",
                    "Multiple Pregnancy",
                    "Intrauterine Foetal Death",
                    "BMI", "Age (25-35)",
                    "Age (30-35)", "Age (35+)",
                    "Haemoglobin Level")

ggplot(varimpdat, aes(x = reorder(Names, Overall), y = Overall)) +
                  geom_bar(stat = "identity") +
                  geom_col(width = 0.1)+
                  coord_flip() + scale_y_continuous(name="") + theme_bw() +
                  theme(axis.text.x = element_text(face="bold", color= "black",
                  size=8, angle=0), axis.text.y = element_text(face="bold", color= "black",
                  size=8, angle=0)) + labs(x = "", title = "")