library(tidyverse)
library(lme4)
library(rsample)
library(glue)

unzip("data/sgemm_product_dataset.zip", exdir = "data")
df <- read_csv("data/sgemm_product.csv")
df$z0 <- seq_len(dim(df)[1]) - 1
df_long <- df %>% pivot_longer(cols = c('Run1 (ms)', 'Run2 (ms)', 'Run3 (ms)', 'Run4 (ms)'),
                               values_to = "t", names_to = "run_id") %>%
  select(-run_id) %>%
  mutate(t = log(t))
x_cols <- colnames(df_long)[-which(colnames(df_long) %in% c("z0", "t"))]

split_by_z <- FALSE
ncv <- 5
if (split_by_z) {
  split_obj <- vfold_cv(tibble(z0= unique(df_long$z0)), v = ncv)
} else {
  split_obj <- vfold_cv(df_long, v = ncv)
}

res_list <- list()

for (k in seq_len(ncv)) {
  cat(glue("  iteration: {k}"), "\n")
  if (split_by_z) {
    z_train <- analysis(split_obj$splits[[k]])$z0
    z_test <- assessment(split_obj$splits[[k]])$z0
    df_train <- df_long %>% filter(z0 %in% z_train)
    df_test <- df_long %>% filter(z0 %in% z_test)
  } else {
    df_train <- analysis(split_obj$splits[[k]])
    df_test <- assessment(split_obj$splits[[k]])
  }
  
  form <- as.formula(str_c("t ~ ", str_c(str_c(x_cols, collapse = " + "), " + (1 | z0)")))
  start <- Sys.time()
  out <- lmer(form, df_train)
  end <- Sys.time()
  
  sigmas <- as.data.frame(VarCorr(out))
  sig2e_est <- sigmas[2, "vcov"]
  sig2b_est <- sigmas[1, "vcov"]
  
  y_pred <- predict(out, df_test, allow.new.levels = TRUE)
  mse <- mean((df_test$t - y_pred)^2)
  
  res_list[[k]] <- list(
    experiment = k - 1,
    exp_type = "lme4",
    mse = mse,
    sig2e_est = sig2e_est,
    sig2b_est = sig2b_est,
    n_epochs = 0,
    time = end - start
  )
}

res_df <- bind_rows(res_list)

write_csv(res_df, "res_sgemm_lme4.csv")
