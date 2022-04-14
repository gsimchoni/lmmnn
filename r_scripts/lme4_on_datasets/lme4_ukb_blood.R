library(tidyverse)
library(lme4)
library(rsample)
library(recipes)
library(glue)

df <- read_csv("data/ukb_triglyc_cancer.csv")

x_cols <- c('weight', 'height_standing', 'gender', 'age', 'smoking',
            'walking_pace', 'stair_climbing', 'get_up_morning','morning_evening',
            'nap', 'sleepiness', 'dozing', 'tobacco', 'skin_color', 'facial_aging',
            'alive_father', 'alive_mother', 'hand_grip_left', 'hand_grip_right')
r_cols <- c("treatment_id", "operation_id", "diagnosis_id", "cancer_id", "histology_id")

ncv <- 5
split_obj <- vfold_cv(df, v = ncv)

res_list <- list()

for (k in seq_len(ncv)) {
  cat(glue("  iteration: {k}"), "\n")
  df_train <- analysis(split_obj$splits[[k]])
  df_test <- assessment(split_obj$splits[[k]])
  
  rec <- recipe(blood_triglyc ~ ., data = df_train %>% select(blood_triglyc, all_of(x_cols), all_of(r_cols))) %>%
    step_normalize(all_numeric(), -all_of(r_cols)) %>%
    prep(df_train)
  
  df_train <- bake(rec, df_train)
  df_test <- bake(rec, df_test)
  
  form <- as.formula(str_c("blood_triglyc ~ ", str_c(str_c(x_cols, collapse = " + "),
                                               " + (1 | treatment_id) + (1 | operation_id) + (1 | diagnosis_id) + (1 | cancer_id) + (1 | histology_id)")))
  start <- Sys.time()
  out <- lmer(form, df_train)
  end <- Sys.time()
  
  sigmas <- as.data.frame(VarCorr(out))
  sig2e_est <- sigmas[6, "vcov"]
  sig2b_est0 <- sigmas[1, "vcov"]
  sig2b_est1 <- sigmas[2, "vcov"]
  sig2b_est2 <- sigmas[3, "vcov"]
  sig2b_est3 <- sigmas[4, "vcov"]
  sig2b_est4 <- sigmas[5, "vcov"]
  
  y_pred <- predict(out, df_test, allow.new.levels = TRUE)
  mse <- mean((df_test$blood_triglyc - y_pred)^2)
  
  res_list[[k]] <- list(
    experiment = k - 1,
    exp_type = "lme4",
    mse = mse,
    sig2e_est = sig2e_est,
    sig2b_est0 = sig2b_est0,
    sig2b_est1 = sig2b_est1,
    sig2b_est2 = sig2b_est2,
    sig2b_est3 = sig2b_est3,
    sig2b_est4 = sig2b_est4,
    n_epochs = 0,
    time = end - start
  )
}

res_df <- bind_rows(res_list)

write_csv(res_df, "res_ukb_blood_triglyc_lme4.csv")
