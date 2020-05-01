#
# Results and Data Analysis code for Paper
# 
#

# ** set your repo dir
setwd("/home/jack/PycharmProjects/BERT_tuning")

#Librarys
library(ggplot2)
library(RColorBrewer)
library(reshape2)
library(dplyr)
library(tidyr)
library(here)
library(gridExtra)


# Data Importing -----------------------------------------------------------------------------------------------------

# dataset with all variables we're possibly interested in
train_data <- read.csv(here(paste0('data/train.csv')))
test_data <- read.csv(here(paste0('data/test.csv')))

# import and combine bert outputs
read_bert_output <- function(dir){
  total_df = data.frame()
  for (i in 1:10){
    temp <- read.csv(paste0(dir, "trail", i, ".csv")) %>% mutate(trial = i)
    total_df <- rbind(total_df, temp)
  }
  return(total_df)
}

bert_frozen <- read_bert_output(here("data/results/bert_frozen/"))
bert_unfrozen <- read_bert_output(here("data/results/bert_unfrozen/"))
bert_last_unfrozen <- read_bert_output(here("data/results/bert_lasttwo_unfrozen/"))

# import simpler models' results
tdif_data <- read.csv(here(paste0('data/results/tfidf_results.csv'))) %>% mutate(model = "TF-IDF Features")
countvec_data <- read.csv(here(paste0('data/results/countvec_results.csv'))) %>% mutate(model = "Token Count Features")
hashvec_data <- read.csv(here(paste0('data/results/hashvec_results.csv'))) %>% mutate(model = "Token Occurrences Features")

# combined simple model results
simpler_models <- rbind(tdif_data,countvec_data) %>% rbind(., hashvec_data)


# training and test set EDA ------------------------------------------------------------------------------------------------

# TODO

# BERT models ------------------------------------------------------------------------------------------------

# data transforming ===========================================================================

bert_frozen_acc <- bert_frozen %>% mutate(pred = ifelse(prediction < 0.5, 0, 1)) %>%
  mutate(correct = pred == target) %>%
  group_by(trial) %>% summarise(acc = (n() -sum(correct)) / n(),
                                recall = 1 - sum(pred & target) / sum(target),
                                prec = 1 - sum(pred & target) / sum(pred)) %>%
  ungroup() %>% mutate(model = "frozen") %>%
  mutate(f1 = 2 * prec * recall / (prec + recall))

bert_unfrozen_acc <- bert_unfrozen %>% mutate(pred = ifelse(prediction < 0.5, 0, 1)) %>%
  mutate(correct = pred == target) %>%
  group_by(trial) %>% summarise(acc = (n() -sum(correct)) / n(),
                                recall = 1 - sum(pred & target) / sum(target),
                                prec = 1 - sum(pred & target) / sum(pred)) %>%
  ungroup() %>% mutate(model = "unfrozen") %>%
  mutate(f1 = 2 * prec * recall / (prec + recall))

bert_last_unfrozen_acc <- bert_last_unfrozen %>% mutate(pred = ifelse(prediction < 0.5, 0, 1)) %>%
  mutate(correct = pred == target) %>%
  group_by(trial) %>% summarise(acc = (n() -sum(correct)) / n(),
                                recall = 1 - sum(pred & target) / sum(target),
                                prec = 1 - sum(pred & target) / sum(pred)) %>%
  ungroup() %>% mutate(model = "last 2 layers unfrozen") %>%
  mutate(f1 = 2 * prec * recall / (prec + recall))

df_plot <- rbind(bert_frozen_acc, bert_unfrozen_acc) %>% rbind(., bert_last_unfrozen_acc)

# plot accuracies ================================================================================
coul <- colorRampPalette(brewer.pal(4, "Spectral") )(8)[c(3,5,7)]


df_plot %>%
  mutate(model = factor(model, levels = c("frozen", "unfrozen", "last 2 layers unfrozen"))) %>%
  ggplot() +
  geom_boxplot(aes(x=model, y=acc, fill=model)) +
  theme_minimal() +
  scale_fill_manual(values = coul) +
  scale_x_discrete(labels = c("Frozen", "Unfrozen", "Last Two Layers Unfrozen")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), text = element_text(size = 16)) + 
  theme(panel.grid.minor = element_blank(), 
        axis.text = element_text(color = "black"),
        text = element_text(color = "black"),
        legend.position = "none") +
  labs(x = "", y = "Accuracies", title = "BERT Model Accuracies")
  
  
  
# statisitical tests  ====================================================================
t.test(acc ~ model, data = df_plot %>% filter(!model %in% c("last 2 layers unfrozen")))
t.test(acc ~ model, data = df_plot %>% filter(!model %in% c("unfrozen")))



# plot F1 ================================================================================

df_plot %>%
  mutate(model = factor(model, levels = c("frozen", "unfrozen", "last 2 layers unfrozen"))) %>%
  ggplot() +
  geom_boxplot(aes(x=model, y=f1, fill=model)) +
  theme_minimal() +
  scale_fill_manual(values = coul) +
  scale_x_discrete(labels = c("Frozen", "Unfrozen", "Last Two Layers Unfrozen")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), text = element_text(size = 16)) + 
  theme(panel.grid.minor = element_blank(), 
        axis.text = element_text(color = "black"),
        text = element_text(color = "black"),
        legend.position = "none") +
  labs(x = "", y = "F1", title = "BERT Model F1 Scores")




# feature vector models -------------------------------------------------------------------------------------

# plot accuracies ================================================================================
coul <- colorRampPalette(brewer.pal(4, "Spectral") )(8)[c(2,6,8)]


simpler_models %>%
  mutate(model = factor(model, levels = c("Token Occurrences Features", "TF-IDF Features", "Token Count Features"))) %>%
  ggplot() +
  geom_boxplot(aes(x=model, y=accuracy, fill=model)) +
  theme_minimal() +
  scale_fill_manual(values = coul) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), text = element_text(size = 16)) + 
  theme(panel.grid.minor = element_blank(), 
        axis.text = element_text(color = "black"),
        text = element_text(color = "black"),
        legend.position = "none") +
  labs(x = "", y = "Accuracies", title = "Feature Model Accuracies")


  

# plot F1 scores ========================================================================================

coul <- colorRampPalette(brewer.pal(4, "Spectral") )(8)[c(2,6,8)]


simpler_models %>%
  mutate(model = factor(model, levels = c("Token Occurrences Features", "TF-IDF Features", "Token Count Features"))) %>%
  ggplot() +
  geom_boxplot(aes(x=model, y=f1, fill=model)) +
  theme_minimal() +
  scale_fill_manual(values = coul) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), text = element_text(size = 16)) + 
  theme(panel.grid.minor = element_blank(), 
        axis.text = element_text(color = "black"),
        text = element_text(color = "black"),
        legend.position = "none") +
  labs(x = "", y = "F1", title = "Feature Model  F1 Scores")







