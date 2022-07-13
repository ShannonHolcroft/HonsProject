require(haven)
require(dplyr)

pph.data = data.frame(read_dta("PPH_data.dta"))

# Extract demographic characteristics and antepartum factors
pph.antepartum.all = pph.data[, c(1:9, 11:12, 14:32, 50:62, 65:73, 76:79, 81)]
names(pph.antepartum.all)
# Extract partum factors
pph.partum = pph.data[, c(1:2, 32:49, 74:76, 80)]
names(pph.partum)

# Extract raw variables
pph.antepartum.rawfactors = pph.antepartum.all[, -c(30:52, 54:57)]
names(pph.antepartum.rawfactors)
pph.partum.rawfactors = pph.partum[, -c(1, 2, 21, 22)]
names(pph.partum.rawfactors)

# Recalculate BMI from Weight & Height
BMI = function(x, y) {
  return(x/(y^2))
}
newbmi = BMI(pph.antepartum.rawfactors$Mothersweight, pph.antepartum.rawfactors$Mothersheight)

pph.antepartum.rawfactors$bmi1 = newbmi

# EXTRACT VARIABLES OF INTEREST
voi = pph.antepartum.rawfactors[, c(2, 3, 5, 8, 10:11, 14, 16, 20:22, 24, 31)]
colnames(voi) = c("PH", "Age", "ED", "IN", "UB", "RE", "Haem", "MP", "Pre", "AH", "HIV", "AN", "BMI")


################################################################################  

# COMPLETE & INCOMPLETE OBSERVATIONS

# VARIABLES OF INTEREST

n.obs.voi = nrow(voi)
# There are 402 complete cases for our VOI
obs.complete.voi = voi[complete.cases(voi),]
# There are 28 observations (6.97%) with missing values
obs.NA.voi = voi[!complete.cases(voi),]
# No observations have more than 1 missing value
NAcounts.voi = apply(voi, 1, function(x) {length(which(is.na(x)))})
table(NAcounts.voi)
# Variables for which data is missing
NA.voi = apply(voi, 2, function(x) {length(which(is.na(x)))})
NA.voi.list = NA.voi[NA.voi > 0]
NA.voi.obs= lapply(NA.voi.list, function(x) 430 - x)
unlist(NA.voi.obs)

# Observations are missing across 6 variables
NA.voi.prop = lapply(NA.voi, function(x) x/n.obs.voi*100)
NA.voi.comp = lapply(NA.voi.prop, function(x) 100 - x)
unlist(NA.voi.prop)
unlist(NA.voi.comp)

# ALL 51 VARIABLES

allrawfactors = cbind(pph.antepartum.rawfactors, pph.partum.rawfactors)
names(allrawfactors)
ncol(allrawfactors)

n.obs = nrow(allrawfactors)
# 378 complete observations
obs.complete.p = allrawfactors[complete.cases(allrawfactors),]
# There are 52 observations with missing values (12.09% of observations)
obs.with.NA.p = allrawfactors[!complete.cases(allrawfactors),]
# 11 observations have 2 missing values (2.56% of observations)
NAcount.by.obs.p = apply(allrawfactors, 1, function(x) {length(which(is.na(x)))})
table(NAcount.by.obs.p)

NAcount.by.var.p = apply(allrawfactors, 2, function(x) {length(which(is.na(x)))})
# Missing data on 25 variables
length(NAcount.by.var.p[NAcount.by.var.p > 0])

kbl(table(NAcount.by.obs.p), format = "latex", booktabs = T) %>%
  column_spec(1:3, border_left = T, border_right = T) %>%
  kable_styling(latex_options = "hold_position")

# ONLY ANTEPARTUM VARIABLES

# 394 complete observations
obs.complete.a = pph.antepartum.rawfactors[complete.cases(pph.antepartum.rawfactors),]
# There are 41 observations with missing values (9.5% of observations)
obs.with.NA.a = pph.antepartum.rawfactors[!complete.cases(pph.antepartum.rawfactors),]
# 5 observations have 2 missing values (1.16% of observations)
NAcount.by.obs.a = apply(pph.antepartum.rawfactors, 1, function(x) {length(which(is.na(x)))})
table(NAcount.by.obs.a)
# Observations are missing across 12 variables when only antepartum considered
NAcount.by.var.a = apply(pph.antepartum.rawfactors, 2, function(x) {length(which(is.na(x)))})
NAcount.by.var.a[NAcount.by.var.a > 0]

# Considering only antepartum is clinically practical and helps with missing data problem

kbl(table(NAcount.by.obs.a), format = "latex", booktabs = T) %>%
  column_spec(1:3, border_left = T, border_right = T) %>%
  kable_styling(latex_options = "hold_position")

################################################################################