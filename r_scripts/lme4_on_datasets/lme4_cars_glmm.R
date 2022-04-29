library(tidyverse)
library(lme4)
library(rsample)
library(glue)
library(pROC)

df <- read_csv("data/cars_df5.csv")
df$long <- ifelse(df$long >= -1.2, 1, 0)
df$price <- as.vector(scale(df$price))
x_cols <- colnames(df)[-which(colnames(df) %in% c("model_id", "location_id", "lat", "long"))]
check_cols <- c("price", "year", "odometer", "manufacturerbmw", "manufacturerchevrolet", "manufacturerdodge",
                "manufacturerford", "manufacturergmc", "manufacturerhonda",
                "manufacturerjeep", "manufacturernissan", "manufacturerother", "manufacturerram", "conditionexcellent",
                "conditionfair", "conditiongood", "conditionlike_new", "condition_na", "conditionnew", "fueldiesel",
                "fuelelectric", "fuelgas", "fuelhybrid", "fuel_na", "title_statusclean", "title_statuslien",
                "title_statusmissing", "title_status_na", "title_statusparts_only", "title_statusrebuilt",
                "transmissionautomatic", "transmissionmanual", "transmission_na", "drive4wd", "drivefwd",
                "drive_na", "sizecompact", "sizefull_size", "sizemid_size", "size_na", "typebus",
                "typeconvertible", "typecoupe", "typehatchback", "typemini_van", "type_na", "typeoffroad", "typeother", "typepickup",
                "typesedan", "type_suv", "typetruck", "typevan", "paint_colorblack",
                "paint_colorblue", "paint_colorbrown", "paint_colorcustom", "paint_colorgreen", "paint_colorgrey",
                "paint_color_na", "paint_colororange", "paint_colorpurple", "paint_colorred", "paint_colorsilver", "paint_colorwhite")

ncv <- 5
split_obj <- vfold_cv(df, v = ncv)

res_list <- list()

for (k in seq_len(ncv)) {
  cat(glue("  iteration: {k}"), "\n")
  df_train <- analysis(split_obj$splits[[k]])
  df_test <- assessment(split_obj$splits[[k]])
  
  form <- as.formula(str_c("long ~ ", str_c(str_c(check_cols, collapse = " + "), " + (1 | model_id)")))
  start <- Sys.time()
  out <- glmer(form, family = binomial, df_train)
  end <- Sys.time()
  
  sigmas <- as.data.frame(VarCorr(out))
  sig2b_est0 <- sigmas[1, "vcov"]
  
  y_pred <- predict(out, df_test, allow.new.levels = TRUE)
  AUC <- as.numeric(roc(df_test$long, y_pred)$auc)
  
  res_list[[k]] <- list(
    experiment = k - 1,
    exp_type = "lme4",
    auc = AUC,
    sig2b_est0 = sig2b_est0,
    n_epochs = 0,
    time = as.numeric(end - start) * 60
  )
}

res_df <- bind_rows(res_list)

write_csv(res_df, "res_cars_glmm_lme4.csv")
