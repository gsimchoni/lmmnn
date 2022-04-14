library(tidyverse)
library(lme4)
library(rsample)
library(glue)

df <- read_csv("data/spotify_df.csv")

x_cols <- colnames(df)[-which(colnames(df) %in% c("artist_id", "album_id", "playlist_id", "subgenre_id", "danceability",
                                                  'track_id', 'track_artist', 'track_album_id', 'track_album_release_date', 'pl_subgenres', 'playlist_ids'))]

ncv <- 5
split_obj <- vfold_cv(df, v = ncv)

res_list <- list()

for (k in seq_len(ncv)) {
  cat(glue("  iteration: {k}"), "\n")
  df_train <- analysis(split_obj$splits[[k]])
  df_test <- assessment(split_obj$splits[[k]])
  
  form <- as.formula(str_c("danceability ~ ", str_c(str_c(x_cols, collapse = " + "),
                                               " + (1 | artist_id) + (1 | album_id) + (1 | playlist_id) + (1 | subgenre_id)")))
  start <- Sys.time()
  out <- lmer(form, df_train)
  end <- Sys.time()
  
  sigmas <- as.data.frame(VarCorr(out))
  sig2e_est <- sigmas[5, "vcov"]
  sig2b_est0 <- sigmas[1, "vcov"]
  sig2b_est1 <- sigmas[2, "vcov"]
  sig2b_est2 <- sigmas[3, "vcov"]
  sig2b_est3 <- sigmas[4, "vcov"]
  
  y_pred <- predict(out, df_test, allow.new.levels = TRUE)
  y_pred <- ifelse(y_pred < 0, 0, ifelse(y_pred > 1, 1, y_pred))
  mse <- mean((df_test$danceability - y_pred)^2)
  
  res_list[[k]] <- list(
    experiment = k - 1,
    exp_type = "lme4",
    mse = mse,
    sig2e_est = sig2e_est,
    sig2b_est0 = sig2b_est0,
    sig2b_est1 = sig2b_est1,
    sig2b_est2 = sig2b_est2,
    sig2b_est3 = sig2b_est3,
    n_epochs = 0,
    time = end - start
  )
}

res_df <- bind_rows(res_list)

write_csv(res_df, "res_spotify_lme4.csv")
