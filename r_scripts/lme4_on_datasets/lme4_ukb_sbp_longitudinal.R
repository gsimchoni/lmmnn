library(tidyverse)
library(lme4)
library(rsample)
library(recipes)
library(glue)

df <- read_csv("data/ukb_sbp_longitudinal.csv")
df$sbp <- df$sbp/100
df$age <- (df$age - min(df$age)) / (max(df$age) - min(df$age))

x_cols <- colnames(df)[-which(colnames(df) %in% c("id", "assess", "sbp"))]
r_cols <- c("id", "age")

pred_future <- FALSE
if (pred_future) {
  df <- df %>% arrange(age)
  n_train <- floor(0.8 * nrow(df))
  df_test_copy <- df[(n_train + 1):nrow(df),]
  df <- df[1:n_train,]
  
}

ncv <- 5
split_obj <- vfold_cv(df, v = ncv)

res_list <- list()

for (k in seq_len(ncv)) {
  cat(glue("  iteration: {k}"), "\n")
  df_train <- analysis(split_obj$splits[[k]])
  if (pred_future) {
    df_test <- df_test_copy
  } else {
    df_test <- assessment(split_obj$splits[[k]])
  }
  
  rec <- recipe(sbp ~ ., data = df_train %>% select(sbp, all_of(x_cols), id)) %>%
    step_normalize(all_numeric(), -all_of(r_cols), -sbp) %>%
    prep(df_train)
  
  df_train <- bake(rec, df_train)
  df_test <- bake(rec, df_test)

  form <- as.formula(str_c("sbp ~ ", str_c(str_c(x_cols, collapse = " + "),
                                               " + (1 |id) + (0 + age|id)")))
  start <- Sys.time()
  out <- lmer(form, df_train)
  end <- Sys.time()
  
  sigmas <- as.data.frame(VarCorr(out))
  sig2e_est <- sigmas[3, "vcov"]
  sig2b_est0 <- sigmas[1, "vcov"]
  sig2b_est1 <- sigmas[2, "vcov"]
  
  y_pred <- predict(out, df_test, allow.new.levels = TRUE)
  mse <- mean((df_test$sbp - y_pred)^2)
  
  res_list[[k]] <- list(
    experiment = k - 1,
    exp_type = "lme4",
    mse = mse,
    sig2e_est = sig2e_est,
    sig2b_est0 = sig2b_est0,
    sig2b_est1 = sig2b_est1,
    n_epochs = 0,
    time = end - start
  )
}

(res_df <- bind_rows(res_list))

if (pred_future) {
  write_csv(res_df, "res_ukb_sbp_longitudinal_lme4_future_scale100_sig2bs2.csv")
} else {
  write_csv(res_df, "res_ukb_sbp_longitudinal_lme4_random_scale100_sig2bs2.csv")
}

