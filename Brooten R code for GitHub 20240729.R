#brooten prediction modeling for mortality in the ed

#load libraries
library(naniar)
library(missForest)
library(randomForest)
library(ROCR)
library(cvAUC)
library(tidyverse)
library(parallel)
library(varSelRF)
library(tableone)
library(caret)
library(missRanger)
library(ranger)
library(glmnet)
library(pmsampsize)
library(sjPlot)
library(sjmisc)
library(sjlabelled)
library(CalibrationCurves)
library(CORElearn)
library(doBy)


#sample size calculation
pmsampsize(type="b",csrsquared=0.1, parameters=40,prevalence=0.02)

#read in data
ds1<-read.csv("Brooten Rdata Predictors 20221111.csv")
dsextra<-read.csv("Brooten Rdata Extra 20221111.csv")

#check that it read in right
str(ds1)
str(dsextra)
summary(ds1)
summary(dsextra)
dim(ds1)

#fix data as necessary
ds1$death<-as.factor(ds1$death)
#5 sex values are missing but are not showing up as NAs, so this turns them to NAs
ds1$SEX<-as.factor(ds1$SEX)
#ds1 %>% replace_with_na_at(.vars=c("SEX"),condition=~.x==.)
ds1$SEX[ds1$SEX == "."] <- NA
ds1<-droplevels(ds1)
summary(ds1$SEX)
#make integer variables into numeric
ds1[sapply(ds1, is.integer)] <- lapply(ds1[sapply(ds1, is.integer)], as.numeric)

#assess missing predictor data
vis_miss(ds1, warn_large_data=FALSE)
gg_miss_var(ds1)

#add shadow matrix to prediction dataset for missing values
ds2<-bind_shadow(ds1)
miss_case_table(ds2)
miss_var_summary(ds2)

#make table overall and by death
paste(names(ds1),collapse="','")
allvarnames<-c('SEX','TEMP','RESP','WEIGHT_LB','SPO2','SPO2_PULSE_RATE','MAP','BMI','PULSE','GCS_SCORE','WBC_VAL','RBC_VAL','HCT_VAL','MCV_VAL','MCH_VAL','RDW_VAL','PLT_VAL','MPV_VAL','NA_VAL','K_VAL','CL_VAL','C02_VAL','BUN_VAL','GLU_VAL','CR_VAL','PROT_VAL','ALB_VAL','BILITOT_VAL','ALKPHOS_VAL','AST_VAL','ALT_VAL','AGAP_VAL','age','sbp','dbp','death')
catvarnames<-c('SEX','death')

##overall
table1overall<-CreateTableOne(vars=allvarnames,data=ds1,factorVars=catvarnames,includeNA = TRUE)
table1overall_print<-print(table1overall,noSpaces=TRUE,printTOggle=FALSE)
write.csv(table1overall_print,"Table 1 Overall 20221111.csv")
print(summary(table1overall))

##by death
table1group<-CreateTableOne(vars=allvarnames,data=ds1,factorVars=catvarnames,strata=c('death'),includeNA = TRUE)
table1group_print<-print(table1group,noSpaces=TRUE,printTOggle=FALSE)
write.csv(table1group_print,"Table 1 by Death 20221111.csv")

#split dataset into test, training
set.seed(7687)
samp_size<-floor(0.7*nrow(ds2))
train_ind<-sample(seq_len(nrow(ds2)),size=samp_size)
ds2$death<-factor(ds2$death)
#delete the patient id and encounter id from this, don't need for imputation
train<-ds2[train_ind,-c(1,3)]
test<-ds2[-train_ind,-c(1,3)]
train$intraindata<-rep(1,length(train[,1]))
test$intraindata<-rep(0,length(test[,1]))
ds_all<-data.frame(rbind(test,train))

#make table comparing test and train data
allvarnames<-c('SEX','TEMP','RESP','WEIGHT_LB','SPO2','SPO2_PULSE_RATE','MAP','BMI','PULSE','GCS_SCORE','WBC_VAL','RBC_VAL','HCT_VAL','MCV_VAL','MCH_VAL','RDW_VAL','PLT_VAL','MPV_VAL','NA_VAL','K_VAL','CL_VAL','C02_VAL','BUN_VAL','GLU_VAL','CR_VAL','PROT_VAL','ALB_VAL','BILITOT_VAL','ALKPHOS_VAL','AST_VAL','ALT_VAL','AGAP_VAL','age','sbp','dbp','death')
catvarnames<-c('SEX','death')
table1datasplit<-CreateTableOne(vars=allvarnames,data=ds_all,factorVars=catvarnames,strata=c('intraindata'),includeNA = TRUE)
table1datasplit_print<-print(table1datasplit,noSpaces=TRUE,printTOggle=FALSE)
write.csv(table1datasplit_print,"Table 2 Comparing Train and Test Datasets 20221111.csv")

#impute missing values in test, training
train_imp<-missRanger(train,pmm.k=10,num.trees=100)
summary(train_imp)
test_imp<-missRanger(test,pmm.k=10,num.trees=100)
summary(test_imp)

write.csv(ds_all, "All train and test data not imputed 20221111.csv")
write.csv(train_imp, "Train data imputed 20221111.csv")
write.csv(test_imp, "Test data imputed 20221111.csv")


###########################################################################
#start here for modeling

#read in saved data from above
#ds_all<-read.csv("All train and test data not imputed 20221111.csv")
#train<-read.csv("Train data imputed 20221111.csv")
#test<-read.csv("Test data imputed 20221111.csv")

train<-train_imp
test<-test_imp

str(ds_all)
str(train)
str(test)
dim(train)

summary(train)
summary(test)

miss_case_table(train)
miss_var_summary(train)
miss_case_table(test)
miss_var_summary(test)
miss_case_table(ds_all)
miss_var_summary(ds_all)

##########logistic regression model with everything
logmodel<-glm(death~.,family=binomial(link='logit'),data=train[,-c(1,38:76)])
tab_model(logmodel,transform = NULL,digits=5)

#evaluate with train data
predprobs<-predict(logmodel,train,type='response')
obs<-factor(train$death)
head(predprobs)
pred<-prediction(predprobs,obs)
opt.cut = function(perf, pred){
    cut.ind = mapply(FUN=function(x, y, p){
        d = (x - 0)^2 + (y-1)^2
        ind = which(d == min(d))
        c(sensitivity = y[[ind]], specificity = 1-x[[ind]], 
            cutoff = p[[ind]])
    }, perf@x.values, perf@y.values, pred@cutoffs)
}
print(opt.cut(performance(pred,"tpr","fpr"), pred))
#cutoff is 0.01893583
expected<-ifelse(predprobs<0.01893583,0,1)
confuse<-table(expected,obs)
confuse
acc<-(confuse[1,1]+confuse[2,2])/sum(confuse)
acc
perf2<-performance(pred,"auc")
auc<-as.numeric(perf2@y.values)
auc
ci.cvAUC(predprobs,obs)
#ci for spec
binom.test(confuse[1,1],confuse[1,1]+confuse[2,1])
#ci for sens
binom.test(confuse[2,2],confuse[1,2]+confuse[2,2])
#ci for NPV
binom.test(confuse[1,1],confuse[1,1]+confuse[1,2])
#ci for PPV
binom.test(confuse[2,2],confuse[2,1]+confuse[2,2])
#pr auc
perf3<-performance(pred,"aucpr")
prauc<-as.numeric(perf3@y.values)
prauc
#brier and CI
brier<-mean((predprobs-as.numeric(obs)+1)^2)
brier

#evaluate with test data
predprobs<-predict(logmodel,newdata=test,type='response')
obs<-factor(test$death)
pred<-prediction(predprobs,obs)
#print(opt.cut(performance(pred,"tpr","fpr"), pred))
#cutoff is 0.01893583
expected<-ifelse(predprobs<0.01893583,0,1)
confuse<-table(expected,obs)
confuse
sum(confuse)
acc<-(confuse[1,1]+confuse[2,2])/sum(confuse)
acc
pred<-prediction(predprobs,obs)
perf2<-performance(pred,"auc")
auc<-as.numeric(perf2@y.values)
auc
ci.cvAUC(predprobs,obs)
#ci for spec
binom.test(confuse[1,1],confuse[1,1]+confuse[2,1])
#ci for sens
binom.test(confuse[2,2],confuse[1,2]+confuse[2,2])
#ci for NPV
binom.test(confuse[1,1],confuse[1,1]+confuse[1,2])
#ci for PPV
binom.test(confuse[2,2],confuse[2,1]+confuse[2,2])
#brier
brier<-mean((predprobs-as.numeric(obs)+1)^2)
brier
#pr auc
perf3<-performance(pred,"aucpr")
prauc<-as.numeric(perf3@y.values)
prauc

#precision recall plot for test data
perf4<-performance(pred,"prec","rec")
plot(perf4)

#calibration plot for test data
val.prob.ci.2(predprobs, (as.numeric(obs)-1), dostats = c("C (ROC)","Brier", "Brier scaled", "Intercept", "Slope", "ECI"),logistic.cal = F)

#recalibrate probabilities with isotonic regression
#function from: https://github.com/easonfg/cali_tutorial/blob/master/lr_iso_recal.R
 train_re_mtx <-cbind(y=obs,yhat=predprobs)
 iso_train_mtx = train_re_mtx[order(train_re_mtx[,2]),]
# create calibration model
  calib.model <- isoreg(iso_train_mtx[,2], iso_train_mtx[,1])
  stepf_data = cbind(calib.model$x, calib.model$yf)
  step_func = stepfun(stepf_data[,1], c(0,stepf_data[,2]))  
# recalibrate and measure on test set
  exp2_iso_recal <- step_func(predprobs)-1
#plot recalibrated data
val.prob.ci.2(exp2_iso_recal, (as.numeric(obs)-1), dostats = c("C (ROC)","Brier", "Brier scaled", "Intercept", "Slope", "ECI"),logistic.cal = F)

#redoing performance stats with recalibrated probs
recalprobs<-exp2_iso_recal
pred<-prediction(recalprobs,obs)
expected<-ifelse(recalprobs<0.01893583,0,1)
confuse<-table(expected,obs)
confuse
sum(confuse)
acc<-(confuse[1,1]+confuse[2,2])/sum(confuse)
acc
pred<-prediction(recalprobs,obs)
perf2<-performance(pred,"auc")
auc<-as.numeric(perf2@y.values)
auc
ci.cvAUC(recalprobs,obs)
#ci for spec
binom.test(confuse[1,1],confuse[1,1]+confuse[2,1])
#ci for sens
binom.test(confuse[2,2],confuse[1,2]+confuse[2,2])
#ci for NPV
binom.test(confuse[1,1],confuse[1,1]+confuse[1,2])
#ci for PPV
binom.test(confuse[2,2],confuse[2,1]+confuse[2,2])
#brier
brier<-mean((recalprobs-as.numeric(obs)+1)^2)
brier
#pr auc
perf3<-performance(pred,"aucpr")
prauc<-as.numeric(perf3@y.values)
prauc







##########logistic regression model without variables with missing values >20%
logmodel_sens<-glm(death~.,family=binomial(link='logit'),data=train[,-c(1,27:33,38:76)])

#evaluate with train data
predprobs<-predict(logmodel_sens,train,type='response')
obs<-factor(train$death)
head(predprobs)
pred<-prediction(predprobs,obs)
opt.cut = function(perf, pred){
    cut.ind = mapply(FUN=function(x, y, p){
        d = (x - 0)^2 + (y-1)^2
        ind = which(d == min(d))
        c(sensitivity = y[[ind]], specificity = 1-x[[ind]], 
            cutoff = p[[ind]])
    }, perf@x.values, perf@y.values, pred@cutoffs)
}
print(opt.cut(performance(pred,"tpr","fpr"), pred))
#cutoff is 0.02059038
expected<-ifelse(predprobs<0.02059038,0,1)
confuse<-table(expected,obs)
confuse
acc<-(confuse[1,1]+confuse[2,2])/sum(confuse)
acc
perf2<-performance(pred,"auc")
auc<-as.numeric(perf2@y.values)
auc
ci.cvAUC(predprobs,obs)
#ci for spec
binom.test(confuse[1,1],confuse[1,1]+confuse[2,1])
#ci for sens
binom.test(confuse[2,2],confuse[1,2]+confuse[2,2])
#ci for NPV
binom.test(confuse[1,1],confuse[1,1]+confuse[1,2])
#ci for PPV
binom.test(confuse[2,2],confuse[2,1]+confuse[2,2])
#pr auc
perf3<-performance(pred,"aucpr")
prauc<-as.numeric(perf3@y.values)
prauc
#brier and CI
brier<-mean((predprobs-as.numeric(obs)+1)^2)
brier

#evaluate with test data
predprobs<-predict(logmodel_sens,newdata=test,type='response')
obs<-factor(test$death)
pred<-prediction(predprobs,obs)
#print(opt.cut(performance(pred,"tpr","fpr"), pred))
#cutoff is 0.01893583
expected<-ifelse(predprobs<0.02059038,0,1)
confuse<-table(expected,obs)
confuse
sum(confuse)
acc<-(confuse[1,1]+confuse[2,2])/sum(confuse)
acc
pred<-prediction(predprobs,obs)
perf2<-performance(pred,"auc")
auc<-as.numeric(perf2@y.values)
auc
ci.cvAUC(predprobs,obs)
#ci for spec
binom.test(confuse[1,1],confuse[1,1]+confuse[2,1])
#ci for sens
binom.test(confuse[2,2],confuse[1,2]+confuse[2,2])
#ci for NPV
binom.test(confuse[1,1],confuse[1,1]+confuse[1,2])
#ci for PPV
binom.test(confuse[2,2],confuse[2,1]+confuse[2,2])
#brier
brier<-mean((predprobs-as.numeric(obs)+1)^2)
#pr auc
perf3<-performance(pred,"aucpr")
prauc<-as.numeric(perf3@y.values)
prauc




##########REMS analysis
#calculate rems and AUC for rems in test data
remsfunction<-function(data){
	remsscore<-0
	if(data$age>=45 & data$age<54) {remsscore<-remsscore+2}
	else if(data$age>=54 & data$age<64) {remsscore<-remsscore+3}
	else if(data$age>=64 & data$age<=74) {remsscore<-remsscore+5}
	else if(data$age>74){remsscore<-remsscore+6}
	if((data$MAP>=110 & data$MAP<130)|(data$MAP>49 & data$MAP<70)) {remsscore<-remsscore+2}
	else if(data$MAP>=130 & data$MAP<=159) {remsscore<-remsscore+3}
	else if((data$MAP>159)|(data$MAP<=49)) {remsscore<-remsscore+4}
	if((data$PULSE>109 & data$PULSE<=139)|(data$PULSE>54 & data$PULSE<=69)) {remsscore<-remsscore+2}
	else if((data$PULSE>139 & data$PULSE<=179)|(data$PULSE>39 & data$PULSE<=54)) {remsscore<-remsscore+3}
	else if((data$PULSE>179)|(data$PULSE<=39)) {remsscore<-remsscore+4}
	if((data$RESP>24 & data$RESP<=34)|(data$RESP>9 & data$RESP<=11)) {remsscore<-remsscore+1}
	else if(data$RESP>5 & data$RESP<=9) {remsscore<-remsscore+2}
	else if((data$RESP>34 & data$RESP<=49)) {remsscore<-remsscore+3}
	else if((data$RESP>49)|(data$RESP<=5)) {remsscore<-remsscore+4}
	if((data$SPO2>=86 & data$SPO2<=89)) {remsscore<-remsscore+1}
	else if((data$SPO2>=75 & data$SPO2<86)) {remsscore<-remsscore+3}
	else if((data$SPO2<75)) {remsscore<-remsscore+4}
	if((data$GCS_SCORE>=10 & data$GCS_SCORE<=13)) {remsscore<-remsscore+1}
	else if((data$GCS_SCORE>=7 & data$GCS_SCORE<10)) {remsscore<-remsscore+2}
	else if((data$GCS_SCORE>=5 & data$GCS_SCORE<7)) {remsscore<-remsscore+3}
	else if((data$GCS_SCORE<5)) {remsscore<-remsscore+4}
	return(remsscore)
}

#convert scores into probabilities based on paper cutoffs
for(i in 1:length(test$death)){
	test$rems_score[i]<-remsfunction(test[i,])
	if(test$rems_score[i]<=2){test$rems_prob<-0}
	else if (test$rems_score[i]>=3 & test$rems_score[i]<=5){test$rems_prob[i]<-0.01}
	else if (test$rems_score[i]>=6 & test$rems_score[i]<=9){test$rems_prob[i]<-0.03}
	else if (test$rems_score[i]>=10 & test$rems_score[i]<=11){test$rems_prob[i]<-0.04}
	else if (test$rems_score[i]>=12 & test$rems_score[i]<=13){test$rems_prob[i]<-0.1}
	else if (test$rems_score[i]>=14 & test$rems_score[i]<=15){test$rems_prob[i]<-0.17}
	else if (test$rems_score[i]>=16 & test$rems_score[i]<=17){test$rems_prob[i]<-0.38}
	else if (test$rems_score[i]>=18 & test$rems_score[i]<=19){test$rems_prob[i]<-0.75}
	else if (test$rems_score[i]>=20 & test$rems_score[i]<=21){test$rems_prob[i]<-0.56}
	else if (test$rems_score[i]>=22 & test$rems_score[i]<=23){test$rems_prob[i]<-0.66}
	else if (test$rems_score[i]>=24 & test$rems_score[i]<=25){test$rems_prob[i]<-1.0}
}

#summarize REMS in our testing data
hist(test$rems_score,main="Histogram of REMS",xlab="REMS Score")
mean(test$rems_score)
sd(test$rems_score)
hist(test$rems_prob,main="Histogram of REMS Predicted Probability",xlab="REMS Pred Probability")
mean(test$rems_prob)
sd(test$rems_prob)

#histogram of REMS score
tiff("Fig 2 Histogram of REMS Score.tiff",res=300,width=5,height=4,units="in")
hist(test$rems_score,main="Histogram of REMS",xlab="REMS Score")
dev.off()

#evaluate performance metrics with test data
predprobs<-test$rems_prob
obs<-factor(test$death)
pred<-prediction(predprobs,obs)

perf2<-performance(pred,"auc")
auc<-as.numeric(perf2@y.values)
auc
ci.cvAUC(predprobs,obs)
#brier
brier<-mean((predprobs-as.numeric(obs)+1)^2)
brier
#pr auc
perf3<-performance(pred,"aucpr")
prauc<-as.numeric(perf3@y.values)
prauc





###############################performance by outcomes
#add shadow matrix to extra dataset for missing values
dsextra2<-bind_shadow(dsextra)
miss_case_table(dsextra2)
miss_var_summary(dsextra2)

#evaluate with test data
predprobs<-predict(logmodel,newdata=test,type='response')
obs<-factor(test$death)
pred<-prediction(predprobs,obs)
#print(opt.cut(performance(pred,"tpr","fpr"), pred))
#cutoff is 0.01893583
expected<-ifelse(predprobs<0.01893583,0,1)
confuse<-table(expected,obs)
confuse

#now we have predicted probs and predicted outcomes for the test set, merge with dsextra
train_ids<-ds2[train_ind,c(1,3)]
test_ids<-ds2[-train_ind,c(1,3)]
train_ids$intraindata<-rep(1,length(train_ids[,1]))
test_ids$intraindata<-rep(0,length(test_ids[,1]))
ds_all_ids<-data.frame(rbind(test_ids,train_ids))
dsextra_all<-merge(dsextra,ds_all_ids,by=c("PAT_MRN_ID","PAT_ENC_CSN_ID"))
dsextra_test1<-dsextra_all[which(dsextra_all$intraindata==0),]
dim(dsextra_test1)
modoutput<-cbind(test_ids,predprobs,expected)
dsextra_test<-merge(dsextra_test1,modoutput,by=c("PAT_MRN_ID","PAT_ENC_CSN_ID"))
summary(dsextra_test)

#make table comparing predicted probabilities by different outcomes
fun <- function(x){
  c(Mean=mean(x), SD=sd(x))
}
r0<-c("Variable","No","Yes","P-value")

vasopressors_p<-summaryBy(predprobs~ VASOPRESSOR_YN, data=dsextra_test, FUN=fun)
r1<-c("Vasopressors",paste0(round(vasopressors_p[1,2],5),"(",round(vasopressors_p[1,3],5),")"),
	paste0(round(vasopressors_p[2,2],5),"(",round(vasopressors_p[2,3],5),")"),
	round(t.test(dsextra_test$predprobs~dsextra_test$VASOPRESSOR_YN)$p.value,5))

vent_p<-summaryBy(predprobs~ ON_VENT, data=dsextra_test, FUN=fun)
r2<-c("Vent",paste0(round(vent_p[1,2],5),"(",round(vent_p[1,3],5),")"),
	paste0(round(vent_p[2,2],5),"(",round(vent_p[2,3],5),")"),
	round(t.test(dsextra_test$predprobs~dsextra_test$ON_VENT)$p.value,5))

death24_p<-summaryBy(predprobs~ death24, data=dsextra_test, FUN=fun)
r3<-c("death24",paste0(round(death24_p[1,2],5),"(",round(death24_p[1,3],5),")"),
	paste0(round(death24_p[2,2],5),"(",round(death24_p[2,3],5),")"),
	round(t.test(dsextra_test$predprobs~dsextra_test$death24)$p.value,5))

death48_p<-summaryBy(predprobs~ death48, data=dsextra_test, FUN=fun)
r4<-c("death48",paste0(round(death48_p[1,2],5),"(",round(death48_p[1,3],5),")"),
	paste0(round(death48_p[2,2],5),"(",round(death48_p[2,3],5),")"),
	round(t.test(dsextra_test$predprobs~dsextra_test$death48)$p.value,5))

death72_p<-summaryBy(predprobs~ death72, data=dsextra_test, FUN=fun)
r5<-c("death72",paste0(round(death72_p[1,2],5),"(",round(death72_p[1,3],5),")"),
	paste0(round(death72_p[2,2],5),"(",round(death72_p[2,3],5),")"),
	round(t.test(dsextra_test$predprobs~dsextra_test$death72)$p.value,5))

death_p<-summaryBy(predprobs~ death, data=dsextra_test, FUN=fun)
r6<-c("death",paste0(round(death_p[1,2],5),"(",round(death_p[1,3],5),")"),
	paste0(round(death_p[2,2],5),"(",round(death_p[2,3],5),")"),
	round(t.test(dsextra_test$predprobs~dsextra_test$death)$p.value,5))

table3probs<-rbind(r0,r1,r2,r3,r4,r5,r6)
table3probs
write.csv(table3probs,"Table 3c Predicted Probability by Outcomes.csv")

#predicted mortality percent from the LR model by outcomes
fun <- function(x){
  c(Mean=mean(x))
}
r0<-c("Variable","No","Yes","P-value")

vasopressors_e<-table(dsextra_test$expected,dsextra_test$VASOPRESSOR_YN)
vasopressors_e
vasopressors_e<-summaryBy(expected~ VASOPRESSOR_YN, data=dsextra_test, FUN=fun)
r1<-c("Vasopressors",paste0(round(vasopressors_e[1,2],5)),paste0(round(vasopressors_e[2,2],5)),
	round(prop.test(x=c(3995,680),n=c(16305+3995,356+680))$p.value,5))

vent_e<-table(dsextra_test$expected,dsextra_test$ON_VENT)
vent_e
vent_e<-summaryBy(expected~ ON_VENT, data=dsextra_test, FUN=fun)
r2<-c("vent",paste0(round(vent_e[1,2],5)),paste0(round(vent_e[2,2],5)),
	round(prop.test(x=c(3809,866),n=c(16165+3809,496+866))$p.value,5))

death24_e<-table(dsextra_test$expected,dsextra_test$death24)
death24_e
death24_e<-summaryBy(expected~ death24, data=dsextra_test, FUN=fun)
r3<-c("death24",paste0(round(death24_e[1,2],5)),paste0(round(death24_e[2,2],5)),
	round(prop.test(x=c(4623,52),n=c(16657+4623,52+4))$p.value,5))

death48_e<-table(dsextra_test$expected,dsextra_test$death48)
death48_e
death48_e<-summaryBy(expected~ death48, data=dsextra_test, FUN=fun)
r4<-c("death48",paste0(round(death48_e[1,2],5)),paste0(round(death48_e[2,2],5)),
	round(prop.test(x=c(4574,101),n=c(4574+16647,101+14))$p.value,5))

death72_e<-table(dsextra_test$expected,dsextra_test$death72)
death72_e
death72_e<-summaryBy(expected~ death72, data=dsextra_test, FUN=fun)
r5<-c("death72",paste0(round(death72_e[1,2],5)),paste0(round(death72_e[2,2],5)),
	round(prop.test(x=c(4540,135),n=c(16640+4540,135+21))$p.value,5))

death_e<-table(dsextra_test$expected,dsextra_test$death)
death_e
death_e<-summaryBy(expected~ death, data=dsextra_test, FUN=fun)
r6<-c("death",paste0(round(death_e[1,2],5)),paste0(round(death_e[2,2],5)),
	round(prop.test(x=c(4277,398),n=c(16572+4277,89+398))$p.value,5))

table3expected<-rbind(r0,r1,r2,r3,r4,r5,r6)
table3expected
write.csv(table3expected,"Table 3d Predicted Death from Model by Outcomes.csv")






###############subgroup analysis
#####age
test_ageLess65<-test[which(test$age<65),]
dim(test_ageLess65)
#evaluate with test data
predprobs<-predict(logmodel,newdata=test_ageLess65,type='response')
obs<-factor(test_ageLess65$death)
pred<-prediction(predprobs,obs)
expected<-ifelse(predprobs<0.01893583,0,1)
confuse<-table(expected,obs)
perf2<-performance(pred,"auc")
auc<-as.numeric(perf2@y.values)
auc
ci.cvAUC(predprobs,obs)
#ci for spec
binom.test(confuse[1,1],confuse[1,1]+confuse[2,1])
sum(confuse)
#ci for sens
binom.test(confuse[2,2],confuse[1,2]+confuse[2,2])
#ci for NPV
binom.test(confuse[1,1],confuse[1,1]+confuse[1,2])
#ci for PPV
binom.test(confuse[2,2],confuse[2,1]+confuse[2,2])
#brier
brier<-mean((predprobs-as.numeric(obs)+1)^2)
brier
#pr auc
perf3<-performance(pred,"aucpr")
prauc<-as.numeric(perf3@y.values)
prauc

test_ageGreater65<-test[which(test$age>=65),]
dim(test_ageGreater65)
#evaluate with test data
predprobs<-predict(logmodel,newdata=test_ageGreater65,type='response')
obs<-factor(test_ageGreater65$death)
pred<-prediction(predprobs,obs)
expected<-ifelse(predprobs<0.01893583,0,1)
confuse<-table(expected,obs)
perf2<-performance(pred,"auc")
auc<-as.numeric(perf2@y.values)
auc
ci.cvAUC(predprobs,obs)
#ci for spec
binom.test(confuse[1,1],confuse[1,1]+confuse[2,1])
#ci for sens
binom.test(confuse[2,2],confuse[1,2]+confuse[2,2])
#ci for NPV
binom.test(confuse[1,1],confuse[1,1]+confuse[1,2])
#ci for PPV
binom.test(confuse[2,2],confuse[2,1]+confuse[2,2])
#brier
brier<-mean((predprobs-as.numeric(obs)+1)^2)
brier
#pr auc
perf3<-performance(pred,"aucpr")
prauc<-as.numeric(perf3@y.values)
prauc

#####sex
test_female<-test[which(test$SEX=="Female"),]
dim(test_female)
#evaluate with test data
predprobs<-predict(logmodel,newdata=test_female,type='response')
obs<-factor(test_female$death)
pred<-prediction(predprobs,obs)
expected<-ifelse(predprobs<0.01893583,0,1)
confuse<-table(expected,obs)
perf2<-performance(pred,"auc")
auc<-as.numeric(perf2@y.values)
auc
ci.cvAUC(predprobs,obs)
#ci for spec
binom.test(confuse[1,1],confuse[1,1]+confuse[2,1])
sum(confuse)
#ci for sens
binom.test(confuse[2,2],confuse[1,2]+confuse[2,2])
#ci for NPV
binom.test(confuse[1,1],confuse[1,1]+confuse[1,2])
#ci for PPV
binom.test(confuse[2,2],confuse[2,1]+confuse[2,2])
#brier
brier<-mean((predprobs-as.numeric(obs)+1)^2)
brier
#pr auc
perf3<-performance(pred,"aucpr")
prauc<-as.numeric(perf3@y.values)
prauc

test_male<-test[which(test$SEX=="Male"),]
dim(test_male)
#evaluate with test data
predprobs<-predict(logmodel,newdata=test_male,type='response')
obs<-factor(test_male$death)
pred<-prediction(predprobs,obs)
expected<-ifelse(predprobs<0.01893583,0,1)
confuse<-table(expected,obs)
perf2<-performance(pred,"auc")
auc<-as.numeric(perf2@y.values)
auc
ci.cvAUC(predprobs,obs)
#ci for spec
binom.test(confuse[1,1],confuse[1,1]+confuse[2,1])
#ci for sens
binom.test(confuse[2,2],confuse[1,2]+confuse[2,2])
#ci for NPV
binom.test(confuse[1,1],confuse[1,1]+confuse[1,2])
#ci for PPV
binom.test(confuse[2,2],confuse[2,1]+confuse[2,2])
#brier
brier<-mean((predprobs-as.numeric(obs)+1)^2)
brier
#pr auc
perf3<-performance(pred,"aucpr")
prauc<-as.numeric(perf3@y.values)
prauc






