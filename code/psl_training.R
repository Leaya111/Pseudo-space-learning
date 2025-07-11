# Pseudo-space Learning (PSL) Model 
# Author: Xinyue Yang


# Load required libraries
library(caret)
library(dplyr)
library(ggplot2)
library(sf)
library(sp)
library(gstat)
library(automap)
library(xgboost)

set.seed(12345)  # Set seed for reproducibility

# Set working directory to current script directory
script_file_path <- rstudioapi::getSourceEditorContext()$path
script_dir <- dirname(script_file_path)
setwd(script_dir)

# Step 1: Read shapefile and extract attribute table
shp_path <- "./SA2_point.shp"  # Replace with your shapefile name in current directory
cat("Reading shapefile...\n")
shp_data <- st_read(shp_path)

cat("Extracting attribute table and filtering missing values...\n")
attribute_table <- st_drop_geometry(shp_data)
attribute_table <- attribute_table[complete.cases(attribute_table), ]

# Step 2: Normalize selected features (excluding target variable 'y')
cat("Normalizing selected columns...\n")
normalize_cols <- colnames(attribute_table)[4:ncol(attribute_table)]
normalize_cols <- setdiff(normalize_cols, "y")
preprocess_model <- preProcess(attribute_table[, normalize_cols], method = "range")
attribute_table[, normalize_cols] <- predict(preprocess_model, attribute_table[, normalize_cols])

# Step 3: Define target variable and features
y_var_all <- "y"
ml_features <- colnames(attribute_table)[c(4:13, 16, 17)]

# Step 4: Create 5-fold cross-validation splits
cat("Creating 5-fold cross-validation splits...\n")
folds <- createFolds(attribute_table[[y_var_all]], k = 5, list = TRUE)
train_test_splits <- list()
for (fold_idx in seq_along(folds)) {
  test_indices <- folds[[fold_idx]]
  train_indices <- setdiff(seq_along(attribute_table[[y_var_all]]), test_indices)
  train_test_splits[[fold_idx]] <- list(train = train_indices, test = test_indices)
}

# Step 5: Define ML model list and pseudo-space combinations
ml_models <- c("rf", "mlp", "svmRadial", "knn", "gbm", "xgbTree")
pseudo_vars <- colnames(attribute_table)[4:17]
two_d_combinations <- combn(pseudo_vars, 2, simplify = FALSE)
three_d_combinations <- combn(pseudo_vars, 3, simplify = FALSE)

# Step 6: Initialize results table
results <- data.frame(Model = character(), Fold = integer(), RMSE = numeric(), Kriging = logical(), stringsAsFactors = FALSE)

# Step 7: Train ML models without kriging
for (model in ml_models) {
  cat(sprintf("Training model: %s\n", model))
  for (fold_idx in seq_along(folds)) {
    train_indices <- train_test_splits[[fold_idx]]$train
    test_indices <- train_test_splits[[fold_idx]]$test
    train_data <- attribute_table[train_indices, c(ml_features, y_var_all)]
    test_data <- attribute_table[test_indices, c(ml_features, y_var_all)]
    
    if (model == "xgbTree") {
      tune_grid <- expand.grid(nrounds = 100, max_depth = 6, eta = 0.1, gamma = 0,
                               colsample_bytree = 0.8, min_child_weight = 1, subsample = 0.8)
      fit <- train(y ~ ., data = train_data, method = model, tuneGrid = tune_grid, trControl = trainControl(method = "none"))
    } else {
      fit <- train(y ~ ., data = train_data, method = model, trControl = trainControl(method = "none"))
    }
    predictions <- predict(fit, newdata = test_data)
    rmse <- sqrt(mean((test_data[[y_var_all]] - predictions)^2))
    results <- rbind(results, data.frame(Model = model, Fold = fold_idx, RMSE = rmse, Kriging = FALSE))
  }
}

# Step 8: Train ML models with 2D kriging residual correction
for (model in ml_models) {
  cat(sprintf("Training 2D kriging model: %s\n", model))
  for (combo in two_d_combinations) {
    for (fold_idx in seq_along(folds)) {
      train_indices <- train_test_splits[[fold_idx]]$train
      test_indices <- train_test_splits[[fold_idx]]$test
      train_data <- attribute_table[train_indices, c(ml_features, y_var_all)]
      test_data <- attribute_table[test_indices, c(ml_features, y_var_all)]
      
      if (model == "xgbTree") {
        tune_grid <- expand.grid(nrounds = 100, max_depth = 6, eta = 0.1, gamma = 0,
                                 colsample_bytree = 0.8, min_child_weight = 1, subsample = 0.8)
        fit <- train(y ~ ., data = train_data, method = model, tuneGrid = tune_grid, trControl = trainControl(method = "none"))
      } else {
        fit <- train(y ~ ., data = train_data, method = model, trControl = trainControl(method = "none"))
      }
      
      train_residuals <- train_data[[y_var_all]] - predict(fit, newdata = train_data)
      train_kriging_data <- data.frame(residuals = train_residuals, x = attribute_table[train_indices, combo[1]], y = attribute_table[train_indices, combo[2]])
      coordinates(train_kriging_data) <- ~ x + y
      
      test_kriging_coords <- data.frame(x = attribute_table[test_indices, combo[1]], y = attribute_table[test_indices, combo[2]])
      coordinates(test_kriging_coords) <- ~ x + y
      
      kriging_result <- autoKrige(residuals ~ 1, input_data = train_kriging_data, new_data = test_kriging_coords)
      corrected_predictions_test <- predict(fit, newdata = test_data) + kriging_result$krige_output@data$var1.pred
      
      rmse <- sqrt(mean((test_data[[y_var_all]] - corrected_predictions_test)^2))
      results <- rbind(results, data.frame(Model = paste(model, "(2D Kriging)"), Fold = fold_idx, RMSE = rmse, Kriging = TRUE))
    }
  }
}

# Step 9: Train ML models with 3D kriging residual correction
for (model in ml_models) {
  cat(sprintf("Training 3D kriging model: %s\n", model))
  best_rmse_per_fold <- rep(Inf, length(folds))
  for (combo in three_d_combinations) {
    var1 <- combo[1]; var2 <- combo[2]; var3 <- combo[3]
    pseudo_space <- data.frame(x = attribute_table[[var1]], y = attribute_table[[var2]], z = attribute_table[[var3]])
    for (fold_idx in seq_along(folds)) {
      train_indices <- train_test_splits[[fold_idx]]$train
      test_indices <- train_test_splits[[fold_idx]]$test
      train_data <- attribute_table[train_indices, c(ml_features, y_var_all)]
      test_data <- attribute_table[test_indices, c(ml_features, y_var_all)]
      
      if (model == "xgbTree") {
        tune_grid <- expand.grid(nrounds = 100, max_depth = 6, eta = 0.1, gamma = 0,
                                 colsample_bytree = 0.8, min_child_weight = 1, subsample = 0.8)
        fit <- train(y ~ ., data = train_data, method = model, tuneGrid = tune_grid, trControl = trainControl(method = "none"))
      } else {
        fit <- train(y ~ ., data = train_data, method = model, trControl = trainControl(method = "none"))
      }
      
      train_residuals <- train_data[[y_var_all]] - predict(fit, newdata = train_data)
      train_kriging_data <- data.frame(residuals = train_residuals, x = pseudo_space[train_indices, "x"], y = pseudo_space[train_indices, "y"], z = pseudo_space[train_indices, "z"])
      coordinates(train_kriging_data) <- ~ x + y + z
      
      test_kriging_coords <- data.frame(x = pseudo_space[test_indices, "x"], y = pseudo_space[test_indices, "y"], z = pseudo_space[test_indices, "z"])
      coordinates(test_kriging_coords) <- ~ x + y + z
      
      kriging_result <- autoKrige(residuals ~ 1, input_data = train_kriging_data, new_data = test_kriging_coords)
      corrected_predictions_test <- predict(fit, newdata = test_data) + kriging_result$krige_output@data$var1.pred
      
      rmse <- sqrt(mean((test_data[[y_var_all]] - corrected_predictions_test)^2))
      if (rmse < best_rmse_per_fold[fold_idx]) {
        best_rmse_per_fold[fold_idx] <- rmse
      }
    }
  }
  for (fold_idx in seq_along(folds)) {
    results <- rbind(results, data.frame(Model = paste(model, "(3D Kriging)"), Fold = fold_idx, RMSE = best_rmse_per_fold[fold_idx], Kriging = TRUE))
  }
}

# Step 10: Plot RMSE results
results <- results %>% distinct(Model, Fold, .keep_all = TRUE)
ggplot(results, aes(x = reorder(Model, RMSE, median), y = RMSE, color = factor(Fold))) +
  geom_boxplot(aes(color = NULL), fill = "gray80", width = 0.4, outlier.shape = NA) +
  geom_jitter(size = 1, width = 0.1, alpha = 0.8) +
  geom_line(aes(group = Fold), alpha = 0.8, size = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(x = "", y = "RMSE", color = "Fold")
