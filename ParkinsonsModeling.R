library(dplyr)
library(caret)

#Read the data
library(readr)
pData <- read_csv("parks_data.csv")
head(pData)

dim(pData)
#195 observations and 24 features

str(pData)
#All numerical features except the 'name' which identifies patient

#'Status' is the target variable.
#Transform the status variable into a factor with 2 levels indicating parkinson's disease or not
pData$status <- as.factor(pData$status)

#Data prep
#Remove the name variable and the id
pData_new <- pData[,-c(1,2)]
str(pData_new)

#Subset the features to exclude the target variable Status
pD_sub <- pData_new[, -which(names(pData_new) == "status")]
str(pD_sub)

cor(pD_sub)

#Subsetting feature sets for PCA
#Jitter and Shimmer variables
MDVP = subset(pData_new, select = c(1:6))
Shimmer_APQ = subset(pData_new, select =c(8:9))
pca_comb = as.data.frame(c(MDVP,Shimmer_APQ))

#Computing Covariance and Eigen values
S <- cov(pD_sub)
sum(diag(S))

#Find Eigen values
s.eigen <- eigen(S)
s.eigen

#PERFORMING PCA on the subsetted data.
pr_comp = prcomp(pca_comb, center = TRUE, scale. = TRUE)
pr_comp

#SUMMARIZE
summary(pr_comp)

#Plotting Principal components with the target labels
library(ggfortify)
pca.plot <- autoplot(pr_comp, data = pData_new, colour = 'status')
pca.plot

#Extracting the principal components into a dataframe
scaling <- pr_comp$sdev[1:2] * sqrt(nrow(pD_sub))
pc1 <- rowSums(t(t(sweep(pD_sub, 2 ,
                         colMeans(pD_sub))) * s.eigen$vectors[,1] * -1) / scaling[1])
pc2 <- rowSums(t(t(sweep(pD_sub, 2 ,
                         colMeans(pD_sub))) * s.eigen$vectors[,2] ) / scaling[2])

#Pooling the PC components into dataframe
df_pca <- data.frame(pc1, pc2, pData_new$status)
names(df_pca)[3] <- "status"

head(df_pca)

df_pca$pData_new.status = NULL

ggplot(df_pca, aes(x=pc1, y=pc2, color=pData_new$status)) + 
  geom_point()


#Training data break up
set.seed(123)
train_size = floor(0.75*nrow(df_pca))
train_ind = sample(seq_len(nrow(df_pca)),size = train_size)

train = df_pca[train_ind,] #creates the training dataset with row numbers stored in train_ind
test = df_pca[-train_ind,]

dim(train)
#146 rows and 4 features

head(train)

dim(test)
#49 rows and 4 features

#RANDOM FOREST MODEL
library(randomForest)
rf_model <- randomForest(status~. ,data=train, importance = TRUE, ntree=100)
summary(rf_model)

#Confusion Matrix
rf_model$confusion

#Importance of Features
importance(rf_model)

#Predictions
p <- predict(rf_model,test)
confusionMatrix(table(p,test$status))

#79% accuracy and 73% Recall achieveed with RandomForest model

#Logistic Regression

library(carat)
logreg <- glm(status~., data=train, family="binomial")
summary(logreg)

pred <- plogis(predict(logreg, test))
pred <- ifelse(pred > 0.5,1,0)

confmat <- table(pred, test$status)
confmat
sum(diag(confmat))/sum(confmat)

confusionMatrix(pred, test$status)
#accuracy = 86%, #Recall = 79%

library(pROC)
roc(train$status, logreg$fitted.values, plot=TRUE)

library(e1071)
#SVM
sv_mod <- svm(formula = status~. , data=train, type="C-classification", kernel="linear", cost=0.1)

sv_mod$coefs
sv_mod$decision.values