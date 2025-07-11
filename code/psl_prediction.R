# psl_prediction.R
# Train pseudo-space learning models and predict with kriging + uncertainty estimation

library(caret)
library(dplyr)
library(ggplot2)
library(sf)
library(automap)
library(sp)
library(gstat)
library(xgboost)

set.seed(12345)

# Normalization function
normalize <- function(x) {
  (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE) + .Machine$double.eps)
}

# Load training data
train_data <- st_read("./SA2_point.shp")
attribute_table <- st_drop_geometry(train_data)
attribute_table <- attribute_table[complete.cases(attribute_table), ]

# Normalize selected columns
normalize_cols <- colnames(attribute_table)[4:ncol(attribute_table)]
for (col in normalize_cols) {
  attribute_table[[col]] <- normalize(attribute_table[[col]])
}

# Define target and features
y_var_all <- "y"
ml_features <- colnames(attribute_table)[c(4:13, 16, 17)]
folds <- createFolds(attribute_table[[y_var_all]], k = 5, list = TRUE)
train_test_splits <- lapply(folds, function(test_idx) {
  list(train = setdiff(seq_len(nrow(attribute_table)), test_idx), test = test_idx)
})

# Initialize best models
best_geo_model <- NULL; best_geo_combo <- c("lon", "lat")
best_attr_model <- NULL; best_attr_combo <- c("popmean", "FP")
best_mix_model <- NULL; best_mix_combo <- c("popmean", "lon", "LST")

# Train MLP in geographical space
cat("Training MLP in geographical space ...\n")
best_score_geo <- Inf
for (i in seq_along(folds)) {
  train_idx <- train_test_splits[[i]]$train
  test_idx <- train_test_splits[[i]]$test
  train_data <- attribute_table[train_idx, c(ml_features, y_var_all)]
  test_data <- attribute_table[test_idx, c(ml_features, y_var_all)]
  
  model <- train(y ~ ., data = train_data, method = "mlp", trControl = trainControl(method = "none"))
  residuals <- train_data[[y_var_all]] - predict(model, newdata = train_data)
  kriging_data <- data.frame(residuals = residuals, x = attribute_table[train_idx, "lon"], y = attribute_table[train_idx, "lat"])
  coordinates(kriging_data) <- ~ x + y
  new_coords <- data.frame(x = attribute_table[test_idx, "lon"], y = attribute_table[test_idx, "lat"])
  coordinates(new_coords) <- ~ x + y
  kr <- autoKrige(residuals ~ 1, kriging_data, new_coords)
  corrected <- predict(model, newdata = test_data) + kr$krige_output@data$var1.pred
  score <- mean(abs(test_data[[y_var_all]] - corrected)) + sqrt(mean((test_data[[y_var_all]] - corrected)^2))
  if (score < best_score_geo) {
    best_geo_model <- model
    train_kriging_data_geo <- kriging_data
  }
}

# Train PSL(MLP) in attribute space
cat("Training PSL(MLP) in attribute space ...\n")
best_score_attr <- Inf
for (i in seq_along(folds)) {
  train_idx <- train_test_splits[[i]]$train
  test_idx <- train_test_splits[[i]]$test
  train_data <- attribute_table[train_idx, c(ml_features, y_var_all)]
  test_data <- attribute_table[test_idx, c(ml_features, y_var_all)]
  
  model <- train(y ~ ., data = train_data, method = "mlp", trControl = trainControl(method = "none"))
  residuals <- train_data[[y_var_all]] - predict(model, newdata = train_data)
  kriging_data <- data.frame(residuals = residuals, x = attribute_table[train_idx, "popmean"], y = attribute_table[train_idx, "FP"])
  coordinates(kriging_data) <- ~ x + y
  new_coords <- data.frame(x = attribute_table[test_idx, "popmean"], y = attribute_table[test_idx, "FP"])
  coordinates(new_coords) <- ~ x + y
  kr <- autoKrige(residuals ~ 1, kriging_data, new_coords)
  corrected <- predict(model, newdata = test_data) + kr$krige_output@data$var1.pred
  score <- mean(abs(test_data[[y_var_all]] - corrected)) + sqrt(mean((test_data[[y_var_all]] - corrected)^2))
  if (score < best_score_attr) {
    best_attr_model <- model
    train_kriging_data_attr <- kriging_data
  }
}

# Train PSL(GBM) in mixed space
cat("Training PSL(GBM) in mixed space...\n")
best_score_mix <- Inf
for (i in seq_along(folds)) {
  train_idx <- train_test_splits[[i]]$train
  test_idx <- train_test_splits[[i]]$test
  train_data <- attribute_table[train_idx, c(ml_features, y_var_all)]
  test_data <- attribute_table[test_idx, c(ml_features, y_var_all)]
  
  model <- train(y ~ ., data = train_data, method = "gbm", trControl = trainControl(method = "none"), verbose = FALSE)
  residuals <- train_data[[y_var_all]] - predict(model, newdata = train_data)
  kriging_data <- data.frame(residuals = residuals,
                             x = attribute_table[train_idx, "popmean"],
                             y = attribute_table[train_idx, "lon"],
                             z = attribute_table[train_idx, "LST"])
  coordinates(kriging_data) <- ~ x + y + z
  new_coords <- data.frame(x = attribute_table[test_idx, "popmean"], y = attribute_table[test_idx, "lon"], z = attribute_table[test_idx, "LST"])
  coordinates(new_coords) <- ~ x + y + z
  kr <- autoKrige(residuals ~ 1, kriging_data, new_coords)
  corrected <- predict(model, newdata = test_data) + kr$krige_output@data$var1.pred
  score <- mean(abs(test_data[[y_var_all]] - corrected)) + sqrt(mean((test_data[[y_var_all]] - corrected)^2))
  if (score < best_score_mix) {
    best_mix_model <- model
    train_kriging_data_mix <- kriging_data
  }
}

# Save models
save(best_geo_model, file = "best_geo_kriging_model.RData")
save(best_attr_model, file = "best_attr_kriging_model.RData")
save(best_mix_model, file = "best_mix_kriging_model.RData")
cat("All models saved.\n")

# -------------------------------------------------------------
# Prediction, residuals, uncertainty estimation on new data
# -------------------------------------------------------------

# Load test data
test_data <- st_read("./SA1_point.shp")

# Rename columns
names(test_data)[names(test_data) == "lst"] <- "LST"
names(test_data)[names(test_data) == "shortwave"] <- "shortwavem"
names(test_data)[names(test_data) == "vis"] <- "vismean"
names(test_data)[names(test_data) == "wsa_nir"] <- "nir"
names(test_data)[names(test_data) == "fp"] <- "FP"
names(test_data)[names(test_data) == "far"] <- "FAR"
names(test_data)[names(test_data) == "avg_diff"] <- "avg_hdiff"

# Prepare and normalize
new_attr <- st_drop_geometry(test_data)
new_attr <- new_attr[complete.cases(new_attr), ]
used_cols <- unique(c(best_geo_combo, best_attr_combo, best_mix_combo, colnames(new_attr)[4:17]))
for (col in used_cols) {
  new_attr[[col]] <- normalize(new_attr[[col]])
}

# Predict + Kriging correction
pred_df <- data.frame(ID = 1:nrow(new_attr))

geo_coords <- new_attr[, best_geo_combo]
coordinates(geo_coords) <- as.formula(paste("~", paste(best_geo_combo, collapse = "+")))
kr_geo <- autoKrige(residuals ~ 1, train_kriging_data_geo, geo_coords)
pred_df$Geo_Kriging_Pred <- predict(best_geo_model, new_attr) + kr_geo$krige_output@data$var1.pred

attr_coords <- new_attr[, best_attr_combo]
coordinates(attr_coords) <- as.formula(paste("~", paste(best_attr_combo, collapse = "+")))
kr_attr <- autoKrige(residuals ~ 1, train_kriging_data_attr, attr_coords)
pred_df$Attr_Kriging_Pred <- predict(best_attr_model, new_attr) + kr_attr$krige_output@data$var1.pred

mix_coords <- new_attr[, best_mix_combo]
coordinates(mix_coords) <- as.formula(paste("~", paste(best_mix_combo, collapse = "+")))
kr_mix <- autoKrige(residuals ~ 1, train_kriging_data_mix, mix_coords)
pred_df$Mix_Kriging_Pred <- predict(best_mix_model, new_attr) + kr_mix$krige_output@data$var1.pred

# True values and residuals
true_y <- new_attr$yy
pred_df$True_Y <- true_y
pred_df$Residual_Geo <- true_y - pred_df$Geo_Kriging_Pred
pred_df$Residual_Attr <- true_y - pred_df$Attr_Kriging_Pred
pred_df$Residual_Mix <- true_y - pred_df$Mix_Kriging_Pred
# Find valid rows (non-missing) used for prediction
valid_rows <- complete.cases(st_drop_geometry(test_data))

# Initialize output fields with NA
test_data$Geo_Kriging_Pred <- NA
test_data$Attr_Kriging_Pred <- NA
test_data$Mix_Kriging_Pred <- NA
test_data$True_Y <- NA
test_data$Residual_Geo <- NA
test_data$Residual_Attr <- NA
test_data$Residual_Mix <- NA

# Assign values only to valid rows
test_data$Geo_Kriging_Pred[valid_rows] <- pred_df$Geo_Kriging_Pred
test_data$Attr_Kriging_Pred[valid_rows] <- pred_df$Attr_Kriging_Pred
test_data$Mix_Kriging_Pred[valid_rows] <- pred_df$Mix_Kriging_Pred
test_data$True_Y[valid_rows] <- pred_df$True_Y
test_data$Residual_Geo[valid_rows] <- pred_df$Residual_Geo
test_data$Residual_Attr[valid_rows] <- pred_df$Residual_Attr
test_data$Residual_Mix[valid_rows] <- pred_df$Residual_Mix

st_write(test_data, "SA1_predicted.shp", delete_layer = TRUE)

# Evaluation metrics
calculate_metrics <- function(true, pred) {
  mae <- mean(abs(true - pred))
  rmse <- sqrt(mean((true - pred)^2))
  r2 <- 1 - sum((true - pred)^2) / sum((true - mean(true))^2)
  return(list(MAE = mae, RMSE = rmse, R2 = r2))
}

m_geo <- calculate_metrics(true_y, pred_df$Geo_Kriging_Pred)
m_attr <- calculate_metrics(true_y, pred_df$Attr_Kriging_Pred)
m_mix <- calculate_metrics(true_y, pred_df$Mix_Kriging_Pred)

cat("Model performance:\n")
cat(sprintf("Geo    : MAE=%.4f RMSE=%.4f R²=%.4f\n", m_geo$MAE, m_geo$RMSE, m_geo$R2))
cat(sprintf("Attr   : MAE=%.4f RMSE=%.4f R²=%.4f\n", m_attr$MAE, m_attr$RMSE, m_attr$R2))
cat(sprintf("Mixed  : MAE=%.4f RMSE=%.4f R²=%.4f\n", m_mix$MAE, m_mix$RMSE, m_mix$R2))

# Uncertainty estimation
uncert_geo <- sqrt(kr_geo$krige_output@data$var1.var) / (pred_df$Geo_Kriging_Pred + .Machine$double.eps)
uncert_attr <- sqrt(kr_attr$krige_output@data$var1.var) / (pred_df$Attr_Kriging_Pred + .Machine$double.eps)
uncert_mix <- sqrt(kr_mix$krige_output@data$var1.var) / (pred_df$Mix_Kriging_Pred + .Machine$double.eps)

test_data$Geo_Kriging_Uncertainty <- NA
test_data$Attr_Kriging_Uncertainty <- NA
test_data$Mix_Kriging_Uncertainty <- NA

test_data$Geo_Kriging_Uncertainty[valid_rows] <- uncert_geo
test_data$Attr_Kriging_Uncertainty[valid_rows] <- uncert_attr
test_data$Mix_Kriging_Uncertainty[valid_rows] <- uncert_mix
st_write(test_data, "SA1_predicted_with_uncertainty.shp", delete_layer = TRUE)

# Density plots
plot_density <- function(data, title, fill_color) {
  ggplot(data.frame(Uncertainty = data[data >= 0 & data <= 1]), aes(x = Uncertainty)) +
    geom_density(fill = fill_color, alpha = 0.6) +
    labs(title = title, x = "Uncertainty", y = "Density") +
    xlim(0, 1) +
    theme_classic(base_size = 14) +
    theme(panel.border = element_rect(color = "black", fill = NA, linewidth = 0.8))
}

print(plot_density(uncert_geo, "Geographical", "#66c2a5"))
print(plot_density(uncert_attr, "2D Pseudo-space", "#fc8d62"))
print(plot_density(uncert_mix, "3D Pseudo-space", "#8da0cb"))
