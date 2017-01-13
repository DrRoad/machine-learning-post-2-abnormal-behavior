# delete these installation lines after 1st run
install.packages("caret")
install.packages("funModeling")
install.packages("Rtsne")

## download data from github, if you have any problem go directly to github: https://github.com/auth0/machine-learning-post-2-abnormal-behavior 
url_git_data = "https://raw.github.com/auth0/machine-learning-post-2-abnormal-behavior/master/data_abnormal.txt"
download.file(githubURL,"data_abnormal.txt")

library(caret)
library(funModeling)
library(Rtsne)

## reading source data
data=read.delim("data_abnormal.txt", header = T, stringsAsFactors = F, sep = "\t")

colnames(data)

freq(data, "abnormal")


################################################  
## MODEL CREATION:
################################################
set.seed(999)
## Setting the validation metric: Cross-Validation 4-fold.
fitControl = trainControl(method = "cv",
                           number = 4,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)

## Creating the model, given the cross-validation method.
fit_model = train(abnormal ~ var_1 + var_2 + var_3 + var_4 + var_5 + var_6 + var_7 + var_8,
                 data = data,
                 method = "rf",
                 trControl = fitControl,
                 verbose = FALSE,
                 metric = "ROC")



data$score=predict(fit_model$finalModel, type="prob")[,2]

# filtering cases...
data_no_abnormal=subset(data, abnormal=="no")

# obtaining the score to filter top 2%
cutoff=quantile(data_no_abnormal$score, probs = c(0.98))

# filtering most suspicious cases
data_to_inspect=subset(data_no_abnormal, score>cutoff)

head(data_to_inspect$id)

# excluding id column and score variable
data_2=data[,!(names(data) %in% c("id", "score"))]
d_dummy = dummyVars(" ~ .", data = data_2)
data_tsne = data.frame(predict(d_dummy, newdata = data))

## creating Rtsne model
set.seed(999)
tsne_model = Rtsne(as.matrix(data_tsne), check_duplicates=FALSE, pca=TRUE, perplexity=30, theta=0.5, dims=2)
d_tsne = as.data.frame(tsne_model$Y)
d_tsne$abnormal = as.factor(data$abnormal)
d_tsne$score=data$score
d_tsne = d_tsne[order(d_tsne$abnormal),]

## plotting
ggplot(d_tsne, aes(x=V1, y=V2, color=abnormal)) +
  geom_point(size=0.25) +
  guides(colour=guide_legend(override.aes=list(size=6))) +
  xlab("") + ylab("") +
  ggtitle("t-SNE on Abnormal Data") +
  theme_light(base_size=20) +
  theme(axis.text.x=element_blank(),
        axis.text.y=element_blank()) +
  geom_point(data=d_tsne[d_tsne$abnormal=="yes",], color="black", alpha=1,shape=21) +
  geom_point(data=d_tsne[d_tsne$abnormal=="no" & d_tsne$score>=cutoff,], color="blue", alpha=0.8,shape=5, aes(color=Class)) +
  scale_colour_brewer(palette = "Set2")

