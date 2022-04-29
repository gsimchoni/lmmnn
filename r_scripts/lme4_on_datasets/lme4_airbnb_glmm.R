library(tidyverse)
library(lme4)
library(rsample)
library(glue)
library(pROC)

df <- read_csv("data/airbnb_after_ETL.csv")
colnames(df) <- janitor::make_clean_names(colnames(df))
# df$price <- as.vector(scale(df$price))
x_cols <- colnames(df)[-which(colnames(df) %in% c("host_id", "air_conditioning", "x1"))]

ncv <- 5
split_obj <- vfold_cv(df, v = ncv)

res_list <- list()

for (k in seq_len(ncv)) {
  cat(glue("  iteration: {k}"), "\n")
  df_train <- analysis(split_obj$splits[[k]])
  df_test <- assessment(split_obj$splits[[k]])
  
  form <- as.formula(str_c("air_conditioning ~ ", str_c(str_c(x_cols, collapse = " + "), " + (1 | host_id)")))
  start <- Sys.time()
  out <- glmer(form, family = binomial, df_train, control = glmerControl(optCtrl = list(maxfun = 100)))
  end <- Sys.time()
  
  sigmas <- as.data.frame(VarCorr(out))
  sig2b_est0 <- sigmas[1, "vcov"]
  
  y_pred <- predict(out, df_test, allow.new.levels = TRUE)
  AUC <- as.numeric(roc(df_test$air_conditioning, y_pred)$auc)
  
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

write_csv(res_df, "res_airbnb_glmm_lme4.csv")
