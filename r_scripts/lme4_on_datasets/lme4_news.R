library(tidyverse)
library(lme4)
library(rsample)
library(glue)

df <- read_csv("data/news_df2.csv")
df$Facebook <- log(df$Facebook + 1)
x_cols <- colnames(df)[-which(colnames(df) %in% c("title_id", "source_id", "Facebook",
                                                  "GooglePlus", "LinkedIn", "IDLink", "Title",
                                                  "Headline", "Source", "Topic", "PublishDate"))]

ncv <- 5
split_obj <- vfold_cv(df, v = ncv)

res_list <- list()

for (k in seq_len(ncv)) {
  cat(glue("  iteration: {k}"), "\n")
  df_train <- analysis(split_obj$splits[[k]])
  df_test <- assessment(split_obj$splits[[k]])
  
  form <- as.formula(str_c("Facebook ~ ", str_c(str_c(x_cols, collapse = " + "), " + (1 | title_id) + (1 | source_id)")))
  start <- Sys.time()
  out <- lmer(form, df_train)
  end <- Sys.time()
  
  sigmas <- as.data.frame(VarCorr(out))
  sig2e_est <- sigmas[3, "vcov"]
  sig2b_est0 <- sigmas[1, "vcov"]
  sig2b_est1 <- sigmas[2, "vcov"]
  
  y_pred <- predict(out, df_test, allow.new.levels = TRUE)
  mse <- mean((df_test$Facebook - y_pred)^2)
  
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

res_df <- bind_rows(res_list)

write_csv(res_df, "res_news_lme4.csv")
