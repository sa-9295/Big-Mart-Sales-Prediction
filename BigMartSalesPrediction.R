#Big Mart Sales Prediction

#Loading libraries
library(data.table)
library(dplyr)
library(caret)
library(ggplot2)
library(xgboost)
#install.packages('corrplot')
library(corrplot)
#install.packages('cowplot')
library(cowplot)

#Reading the training and test datasets
training_set = fread("Train_UWu5bXk.txt",sep=',')
test_set = fread("Test_u94Q5KV.txt", sep=',')
submission = fread("SampleSubmission_TmnO39y.txt")

#Features of Data
names(training_set)
names(test_set)

#Structure of Data
str(training_set)
str(test_set)

#Combining training and test data for analysis
test_set$Item_Outlet_Sales =NA
dataset = rbind(training_set,test_set)

#Exploratory Data Analysis
#Univariate Analysis

#Item Outlet Sales distribution
ggplot(data = training_set, aes(training_set$Item_Outlet_Sales)) + geom_histogram(
  binwidth = 150,
  fill = "dark green", col="red", alpha=0.3) + xlab("Item_Outlet_Sales")

#Item Weight distribution
p1 = ggplot(data = dataset, aes(dataset$Item_Weight)) + geom_histogram(
  binwidth = 0.5, fill="blue")+xlab("Item_Weight")
#Item Visibility distribution
p2 = ggplot(data = dataset, aes(dataset$Item_Visibility)) + geom_histogram(
  binwidth = 0.005, fill="blue")+xlab("Item_Visibility")
#Item MRP distribution
p3 = ggplot(data = dataset, aes(dataset$Item_MRP)) + geom_histogram(
  binwidth = 1, fill="blue")+xlab("Item_MRP")
plot_grid(p1,p2,p3,nrow=1) #Plotting all 3 graphs

#Item Fat Content distribution
ggplot(dataset %>% group_by(Item_Fat_Content) %>% summarise(Count=n())) +
  geom_bar(aes(Item_Fat_Content, Count), stat ="identity", fill="coral1")

dataset$Item_Fat_Content[dataset$Item_Fat_Content == "LF"] = "Low Fat"
dataset$Item_Fat_Content[dataset$Item_Fat_Content == "low fat"] = "Low Fat"
dataset$Item_Fat_Content[dataset$Item_Fat_Content == "reg"] = "Regular"

ggplot(dataset %>% group_by(Item_Fat_Content) %>% summarise(Count=n())) +
  geom_bar(aes(Item_Fat_Content, Count), stat ="identity", fill="coral1")

#Item Type distribution
p4 = ggplot(dataset %>% group_by(Item_Type) %>% summarise(Count=n())) +
  geom_bar(aes(Item_Type, Count), stat ="identity", fill="coral1") + xlab("") +
  geom_label(aes(Item_Type,Count,label=Count),vjust=0.1) +
  theme(axis.text.x = element_text(angle=45,hjust=1)) + ggtitle("Item_Type")
#Outlet Identifier distribution
p5 = ggplot(dataset %>% group_by(Outlet_Identifier) %>% summarise(Count=n())) +
  geom_bar(aes(Outlet_Identifier, Count), stat ="identity", fill="coral1")  +
  geom_label(aes(Outlet_Identifier,Count,label=Count),vjust=0.1) +
  theme(axis.text.x = element_text(angle=45,hjust=1))
#Outlet Size distribution
p6 = ggplot(dataset %>% group_by(Outlet_Size) %>% summarise(Count=n())) +
  geom_bar(aes(Outlet_Size, Count), stat ="identity", fill="coral1") +
  geom_label(aes(Outlet_Size,Count,label=Count),vjust=0.1) +
  theme(axis.text.x = element_text(angle=45,hjust=1))

second_row = plot_grid(p5,p6,nrow=1)
plot_grid(p4,second_row,ncol=1) #Plotting all together

#Outlet Establishment Year distribution
p7 = ggplot(dataset %>% group_by(Outlet_Establishment_Year) %>% summarise(Count = n())) +   
  geom_bar(aes(factor(Outlet_Establishment_Year), Count), stat = "identity", fill = "coral1") +  
  geom_label(aes(factor(Outlet_Establishment_Year), Count, label = Count), vjust = 0.5) +  xlab("Outlet_Establishment_Year") +  theme(axis.text.x = element_text(size = 8.5))
#Outlet Type distribution
p8 = ggplot(dataset %>% group_by(Outlet_Type) %>% summarise(Count = n())) +   
  geom_bar(aes(Outlet_Type, Count), stat = "identity", fill = "coral1") +  
  geom_label(aes(factor(Outlet_Type), Count, label = Count), vjust = 0.5) +  theme(axis.text.x = element_text(size = 8.5))
plot_grid(p7, p8, nrow = 1) #Plotting all together

#Bivariate Analysis
#Extraxting training data from combined data
training_set=dataset[1:nrow(training_set)]

#Item Weight vs Item Outlet Sales Distribution
p9 = ggplot(training_set) + geom_point(aes(Item_Weight,Item_Outlet_Sales), colour="violet", alpha=0.3) +
  theme(axis.title = element_text(size=8.5))
#Item Visibility vs Item Outlet Sales Distribution
p10 = ggplot(training_set) + geom_point(aes(Item_Visibility,Item_Outlet_Sales), colour="violet", alpha=0.3) +
  theme(axis.title = element_text(size=8.5))
#Item MRP vs Item Outlet Sales Distribution
p11 = ggplot(training_set) + geom_point(aes(Item_MRP,Item_Outlet_Sales), colour="violet", alpha=0.3) +
  theme(axis.title = element_text(size=8.5))

second_row_2 = plot_grid(p10,p11,ncol=2)
plot_grid(p9,second_row_2,nrow=2) #Plotting all together

#Item Type vs Item Outlet Sales Distribution
p12 = ggplot(training_set) + geom_violin(aes(Item_Type, Item_Outlet_Sales), fill = "magenta") +   
  theme(axis.text.x = element_text(angle = 45, hjust = 1),axis.text = element_text(size = 8),
        axis.title = element_text(size = 8.5))
#Item Fat Content vs Item Outlet Sales Distribution
p13 = ggplot(training_set) + geom_violin(aes(Item_Fat_Content, Item_Outlet_Sales), fill = "magenta") +   
  theme(axis.text.x = element_text(angle = 45, hjust = 1),axis.text = element_text(size = 8),
        axis.title = element_text(size = 8.5))
#Outlet Identifier vs Item Outlet Sales Distribution
p14 = ggplot(training_set) + geom_violin(aes(Outlet_Identifier, Item_Outlet_Sales), fill = "magenta") +   
  theme(axis.text.x = element_text(angle = 45, hjust = 1),axis.text = element_text(size = 8),
        axis.title = element_text(size = 8.5))

second_row_3 = plot_grid(p13,p14,ncol=2)
plot_grid(p12,second_row_3,ncol=1) #Plotting all together

#Outlet Size vs Item Outlet Sales Distribution
ggplot(training_set) + geom_violin(aes(Outlet_Size,Item_Outlet_Sales), fill='magenta')
#Distribution of Small Outlet Size is similar to distribution of blank Outlet Size. So, we can
#substitute blank as Small.
# dataset$Outlet_Size=as.character(dataset$Outlet_Size)
# dataset$Outlet_Size[dataset$Outlet_Size==""] = "Small"
# dataset$Outlet_Size = as.factor(dataset$Outlet_Size)

#Outlet Location Type vs Item Outlet Sales Distribution
p15 = ggplot(training_set) + geom_violin(aes(Outlet_Location_Type, Item_Outlet_Sales), fill = "magenta") +   
  theme(axis.text = element_text(size = 8), axis.title = element_text(size = 8.5))
#Outlet Type vs Item Outlet Sales Distribution
p16 = ggplot(training_set) + geom_violin(aes(Outlet_Type, Item_Outlet_Sales), fill = "magenta") +   
  theme(axis.text = element_text(size = 8), axis.title = element_text(size = 8.5))
plot_grid(p15,p16,ncol=1)

#Treating Missing Values
#Checking missing value for Item Weight
sum(is.na(dataset$Item_Weight))
#Impute missing Item Weight with mean weight as per Item Identifier
missing_index = which(is.na(dataset$Item_Weight))
for(i in missing_index){
  item = dataset$Item_Identifier[i]
  dataset$Item_Weight[i] = mean(dataset$Item_Weight[dataset$Item_Identifier == item],na.rm=T)
}
sum(is.na(dataset$Item_Weight)) # Checking

#Impute 0 in Item Visibility by mean as per Item Identifier
zero_index = which(dataset$Item_Visibility == 0)
for(i in zero_index){
  item = dataset$Item_Identifier[i]
  dataset$Item_Visibility[i] = mean(dataset$Item_Visibility[dataset$Item_Identifier == item],na.rm=T)
}
#Plot and check
ggplot(dataset) + geom_histogram(aes(Item_Visibility), fill="blue", bins=100)

#Feature Engineering

#1. Classify Item Type as Perishable and Non-Perishable
perishable = c("Breads", "Breakfast", "Dairy", "Fruits and Vegetables", "Meat", "Seafood")
non_perishable = c("Baking Goods", "Canned", "Frozen Foods", "Hard Drinks", "Health and Hygiene",
                   "Household", "Soft Drinks")
dataset[,Item_Type_new := ifelse(Item_Type %in% perishable, "perishable", 
                                 ifelse(Item_Type %in% non_perishable, "non_perishable","not_sure"))]

#2. Categorizing using first two characters of Item Identifier as DR, FD or NC
table(dataset$Item_Type,substr(dataset$Item_Identifier,1,2))
dataset[,Item_category := substr(dataset$Item_Identifier,1,2)]

#3. Changing value of Item Fat Content wherever Item Category is NC
dataset$Item_Fat_Content[dataset$Item_category == "NC"] = "Non-Edible"

#4. Calculating years of Operation for Outlet
dataset[,Outlet_Years := 2013 - Outlet_Establishment_Year]
dataset$Outlet_Establishment_Year = as.factor(dataset$Outlet_Establishment_Year)

#5. Price per Unit Weight
dataset[, price_per_unit_wt := Item_MRP/Item_Weight]

#6. Item MRP is distributed in four bins with respect to Item Outlet Sales
dataset[, Item_MRP_clusters := ifelse(Item_MRP<69, "1st", ifelse(Item_MRP>=69 & Item_MRP<136, "2nd",
                               ifelse(Item_MRP>=136 & Item_MRP<203, "3rd", "4th")))]

#Encoding Ordinal Variables
dataset[,Outlet_Size_num := ifelse(Outlet_Size=="Small", 0, 
                                   ifelse(Outlet_Size=="Medium",1,2))]

dataset[,Outlet_Location_Type_num := ifelse(Outlet_Location_Type=="Tier 3", 0, 
                                   ifelse(Outlet_Size=="Tier 2",1,2))]

dataset[,c("Outlet_Size","Outlet_Location_Type") := NULL]

#One Hot Encoding for Categorical variables
ohe = dummyVars("~.", data = dataset[,-c("Item_Identifier", "Outlet_Establishment_Year","Item_Type")],
                fullRank=T)
ohe_df = data.table(predict(ohe,dataset[,-c("Item_Identifier", "Outlet_Establishment_Year","Item_Type")]))
dataset=cbind(dataset[,"Item_Identifier"],ohe_df)

#Removing Skewness
dataset[,Item_Visibility := log(Item_Visibility + 1)]
dataset[,price_per_unit_wt := log(price_per_unit_wt + 1)]

#Scaling numerica variables
#First find the index of all numeric variables
num_vars = which(sapply(dataset,is.numeric))
num_var_names = names(num_vars)
dataset_numeric = dataset[,setdiff(num_var_names,"Item_Outlet_Sales"), with = F]
prep_num = preProcess(dataset_numeric, method=c("center","scale"))
dataset_numeric_norm = predict(prep_num,dataset_numeric)
dataset[,setdiff(num_var_names, "Item_Outlet_Sales") := NULL]
dataset = cbind(dataset,dataset_numeric_norm)

#Splitting combined data to training and test set
training_set = dataset[1:nrow(training_set)]
test_set = dataset[(nrow(training_set)+1):nrow(dataset)]
test_set[,Item_Outlet_Sales := NULL]

#Analyzing Correlated Variables
cor_train = cor(training_set[,-c("Item_Identifier")])
corrplot(cor_train, method="pie", type="lower", tl.cex=0.9)

#Model Building

#1. Linear Regression
 regressor = lm(formula = Item_Outlet_Sales~., data = training_set[,-c("Item_Identifier")])

#Predicting the test set results
#submission$Item_Outlet_Sales = predict(regressor, test_set[,-c("Item_Identifier")])
 write.csv(submission, "Linear_Reg_Result.csv", row.names =FALSE)
#Using K fold cross validation
# library(caret)
# folds = createFolds(training_set$Item_Outlet_Sales,k=5)
# cv = lapply(folds,function(x){
#   training_fold = training_set[-x,]
#   test_fold = training_set[x,]
#   regressor = lm(formula = Item_Outlet_Sales~., data = training_fold[,-c("Item_Identifier")])
#   submission$Item_Outlet_Sales = predict(regressor, test_fold[,-c("Item_Identifier")])
# })

#2. Regularized Linear Regression
#Lasso Regression
#install.packages("glmnet")
set.seed(12345)
#library(caret)
library(glmnet)
my_control = trainControl(method="cv", number =5) #Performing 5 fold cross validation
Grid = expand.grid(alpha =1, lambda = seq(0.001,0.1,by=0.0002)) #alpha = 1 for Lasso
training_lasso = as.matrix(training_set[,-c("Item_Identifier", "Item_Outlet_Sales")])
lasso_regressor = train(x=training_lasso,
                        y = training_set$Item_Outlet_Sales, method = "glmnet", 
                        trControl = my_control, tuneGrid=Grid)
#Predicting the test set results
submission$Item_Outlet_Sales = predict(lasso_regressor, test_set[,-c("Item_Identifier")])
write.csv(submission, "Lasso_Reg_Result.csv", row.names =FALSE)

#Ridge Regression
set.seed(1236)
#library(caret)
library(glmnet)
my_control = trainControl(method="cv", number =5) #Performing 5 fold cross validation
Grid = expand.grid(alpha =0, lambda = seq(0.001,0.1,by=0.0002)) #alpha = 0 for Ridge
training_ridge = as.matrix(training_set[,-c("Item_Identifier", "Item_Outlet_Sales")])
ridge_regressor = train(x=training_ridge,
                        y = training_set$Item_Outlet_Sales, method = "glmnet", 
                        trControl = my_control, tuneGrid=Grid)
#Predicting the test set results
submission$Item_Outlet_Sales = predict(lasso_regressor, test_set[,-c("Item_Identifier")])
write.csv(submission, "Ridge_Reg_Result.csv", row.names =FALSE)

#3. Random Forest
set.seed(1237)
#install.packages("ranger")
my_control=trainControl(method="cv",number =5)
tgrid = expand.grid(.mtry=c(3:10), .splitrule = "variance", .min.node.size = c(10,15,20))
rf_regressor = train(x=training_set[,-c("Item_Identifier", "Item_Outlet_Sales")],
                     y=training_set$Item_Outlet_Sales, method="ranger",
                     trControl = my_control, tuneGrid = tgrid,
                     num.trees=400, importance = "permutation")

#Performance
plot(rf_regressor) # We can see best score at mtry = 5 and min.node.size = 20

#Plotting feature Importance
plot(varImp(rf_regressor)) # We can see that Item MRP is the most important variable in predicting
#Predicting target variable
submission$Item_Outlet_Sales = predict(rf_regressor, test_set[,-c("Item_Identifier")])
write.csv(submission, "RandomForest_Result.csv", row.names =FALSE)

#4. Support Vector Regression
library(e1071)
svr_regressor = svm(formula =training_set$Item_Outlet_Sales~., data=training_set[,-c("Item_Identifier", "Item_Outlet_Sales")],
                    type = "eps-regression")
#Predicting target variable
submission$Item_Outlet_Sales = predict(svr_regressor, test_set[,-c("Item_Identifier")])
write.csv(submission, "SVR_Result.csv", row.names =FALSE)

#5. XGBoost
param_list = list(objective="reg:linear", eta=0.01, gamma=1,max_depth=6,subsample=0.8,colsample_bytree=0.5)
dtrain=xgb.DMatrix(data = as.matrix(training_set[,-c("Item_Identifier", "Item_Outlet_Sales")]), 
                       label= training_set$Item_Outlet_Sales) 
dtest = xgb.DMatrix(data = as.matrix(test_set[,-c("Item_Identifier")]))
#Applying cross validation to find optimal values for nrounds
set.seed(122)
xgbcv=xgb.cv(params = param_list, data=dtrain, nrounds=1000, nfold=5, print_every_n = 10,
             early_stopping_rounds = 30, maximize = F)
#Got best result at 425
xgb_regressor = xgb.train(data=dtrain, params=param_list,nrounds = 425)
#Predicting target variable
submission$Item_Outlet_Sales = predict(xgb_regressor, dtest)
write.csv(submission, "XGBoost_Result.csv", row.names =FALSE)
#Variable Importance
var_imp = xgb.importance(feature_names = setdiff(names(training_set),c("Item_Identifier", "Item_Outlet_Sales")),
                         model=xgb_regressor)
xgb.plot.importance(var_imp)

