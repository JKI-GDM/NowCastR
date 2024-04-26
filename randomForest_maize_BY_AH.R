# Script: Supporting script to install packages for Towards regional nowcasting of soil-erosion events on arable land-----------------------------
# Author: Pedro Batista (pedro.batista@geo.uni-augsburg.de)
# Date Created: 2024-04-22
# R version 4.1.1 (2021-08-10)
#

#---- Session Info --------------------------------------
# --- randomForest 4.7-1     --- purrr     0.3.4
# --- foreach      1.5.1     --- readr     2.0.2
# --- caret        6.0-89    --- tidyr     1.4.1 
# --- lattice      0.20-44   --- tibble    3.1.4
# --- forcats      0.5.1     --- ggplot2   3.3.5
# --- stringr      1.4.0     --- tidyverse 1.3.1
# --- dplyr        1.0.7     --- pacman    0.5.1
    

  
# Clean the working environment
rm(list=ls())


# use Pacman library to load or install the rest of packages
if (!require("pacman")) install.packages("pacman")

# Load or install all the rest of packages and functions automatically
p_load(tidyverse, caret, foreach, randomForest)
select <- dplyr::select

# Load maize input data
df_maize <- read.csv("./input_maize.csv") %>%
  as_tibble() 

# Let us detect erosion events per field parcel
# Get sample size for minority class
df_maize %>%
  group_by(Eroding) %>%
  summarise(n = n()) # 173 non-eroded fields

# Define predictors (covariates)
# All columns except target variables
predictors <- colnames(df_maize %>%
                         select(-c(Erosion_class, 
                                   Eroding))
                       )

# Sample minority class 30 times, train random forest model, and summarise results
set.seed(1) # set seed for reproducibility
imp_stats <- foreach(i = 1:30, .combine = rbind) %do%
  {
    # create balanced df from minority sample
    df_balanced <- df_maize %>%
      group_by(Eroding) %>%
      sample_n(round(173)) %>% # 
      mutate(Erosion_class = as.factor(Erosion_class)) %>%
      droplevels() %>%
      ungroup() %>%
      drop_na()
    
    # 4-fold cross validation
    model <- train(df_balanced[,predictors],
                   df_balanced$Eroding,
                   method = "rf",
                   tuneGrid = data.frame("mtry" = sqrt(length(predictors))),
                   importance = TRUE,
                   trControl = trainControl(method = "cv",
                                            number = 4))
    # Get variable importance ranking
    imp <- varImp(model)$importance %>%
      as_tibble(rownames = "variable")
  }

# Summarise results
imp_summ <- imp_stats %>%
  group_by(variable) %>%
  summarise_all(.funs = mean) %>%
  mutate(variable = factor(variable)) %>%
  pivot_longer(!variable) 

# Create a directory to export the results if it does not exist
if (!file.exists("./Results/")) {
  dir.create("./Results/")
}

# Write results to a CSV file
write.csv(imp_summ, 
          "./Results/impRank_table_maize.csv")

# Plot importance ranking of predictors
p1 <- ggplot(imp_summ, aes(x=reorder(variable, value),
                           y = value)) +
  geom_point(colour = "green4") +
  geom_segment(aes(x = variable,
                   xend = variable,
                   y = 0,
                   yend = value)) +
  coord_flip() +
  theme_bw(base_size = 14) +
  theme(panel.grid = element_blank()) +
  labs(y = "Importance", x = "") 

p1

# Save plot
ggsave(
  "./Results/impRank-balanced-4-fold-CV.png", 
  p1, 
width = 15,
height = 10,
units = "cm")

# Same loop and generate confusion matrix
set.seed(1)
cm_stats <- foreach(i = 1:30) %do%
  {
    df_balanced <- df_maize %>%
      group_by(Eroding) %>%
      sample_n(round(173)) %>% # minority sample size
      mutate(Erosion_class = as.factor(Erosion_class)) %>%
      droplevels() %>%
      ungroup() %>%
      drop_na()
    
    model <- train(df_balanced[,predictors],
                   df_balanced$Eroding,
                   method = "rf",
                   tuneGrid = data.frame("mtry" = sqrt(length(predictors))),
                   importance = TRUE,
                   trControl = trainControl(method = "cv",
                                            number = 4))
    
    conf_max <- confusionMatrix(model, norm = "none")$table
  }

# Calculate the mean of matrices in the list
cm_mean <- round(Reduce(`+`, cm_stats) / length(cm_stats))

# View the mean matrix
print(cm_mean)

# Convert confusion matrix to tibble
cm_table <- as.table(cm_mean) %>%
  as_tibble() %>%
  mutate(
    n = round(n),
    Percent = round(n/sum(n) * 100))

# Calculate percentages
cm_table$Percentage <- scales::percent(cm_table$Percent,
                                       scale = 1)

# Plot confusion matrix
p2 <- ggplot(cm_table, 
           aes(Reference, Prediction, fill = n)) +
  geom_tile() + 
  geom_text(aes(label = Percentage)) +
  scale_fill_gradient(
    # trans = "log",
    # trans = "sqrt",
    low="white", 
    high="green4") +
  labs(y = "Prediction",
       x = "Reference",
       fill = "n"
  ) +
  theme_bw(base_size = 14) +
  theme(legend.text = element_blank(),
        axis.title.x = element_text(vjust = 1),
        axis.title.y = element_text(vjust = 2),
        legend.position = "none",
        panel.grid = element_blank()) 

p2

# Save plot
ggsave(
  "./Results/confMatrix-balanced-4-fold-CV.png",
  p2,
  width = 15,
  height = 10,
  units = "cm"
  )

# Write results to a CSV file
write.csv(cm_table,
          "./Results/confMatrix_maize.csv")

# Calculate accuracy, recall, and precision
recall <- diag(cm_mean) / colSums(cm_mean) * 100
precision <- diag(cm_mean) / rowSums(cm_mean) * 100
overall_accuracy <- sum(diag(cm_mean)) / sum(cm_mean) * 100
# Overall accuracy is 72% 

