library(tidyverse)
library(reticulate)
library(lme4)

lmmnn_utils <- import("lmmnn.utils")
params_dict <- dict(n_fixed_effects = 10L, n_per_cat = 3, fixed_intercept = 1,
                    X_non_linear = TRUE, Z_non_linear = FALSE, Z_embed_dim_pct = 10)
mode <- "intercepts"
sig2e <- 1.0
sig2b_list <- list(list(0.5, 5.0), list(0.5, 5.0))
q_list <- list(list(1000L, 10000L), list(1000L, 10000L))
sig2bs_spatial <- NULL
q_spatial <- NULL
N <- 100000L
n_iter <- 5

res_list <- list()
counter <- 0

for (sig2b0 in sig2b_list[[1]]) {
  for (sig2b1 in sig2b_list[[2]]) {
    for (q0 in q_list[[1]]) {
      for (q1 in q_list[[2]]) {
        cat(glue::glue("N: {N}, sig2e: {sig2e}; sig2b0: {sig2b0}; sig2b1: {sig2b1}; q0: {q0}; q1: {q1}"), "\n")
        for (k in seq_len(n_iter)) {
          counter <- counter + 1
          cat(glue::glue("  iteration: {k}"), "\n")
          py_res <- lmmnn_utils$generate_data(
            mode = mode, qs = list(q0, q1), sig2e = sig2e, sig2bs = list(sig2b0, sig2b1),
            sig2bs_spatial = sig2bs_spatial, q_spatial = q_spatial,
            N = N, rhos = list(), p_censor = list(), params = params_dict)
          X_train = py_res[[1]]
          X_test = py_res[[2]]
          y_train = py_res[[3]]
          y_test = py_res[[4]]
          
          df_train <- data.frame(y = y_train)
          df_train <- cbind(df_train, X_train)
          df_test <- data.frame(y = y_test)
          df_test <- cbind(df_test, X_test)
          
          form <- as.formula(str_c("y ~ ", str_c(str_c(str_c("X", 0:9), collapse = " + "), " + (1 | z0)  + (1 | z1)")))
          start <- Sys.time()
          out <- lmer(form, df_train)
          end <- Sys.time()
          
          sigmas <- as.data.frame(VarCorr(out))
          sig2e_est <- sigmas[3, "vcov"]
          sig2b0_est <- sigmas[1, "vcov"]
          sig2b1_est <- sigmas[2, "vcov"]
          
          y_pred <- predict(out, df_test, allow.new.levels = TRUE)
          mse <- mean((y_test - y_pred)^2)
          res_list[[counter]] <- list(
            N = N,
            sig2e = sig2e,
            sig2b0 = sig2b0,
            sig2b1 = sig2b1,
            q0 = q0,
            q1 = q1,
            deep = FALSE,
            experiment = k,
            exp_type = "lme4",
            mse = mse,
            sig2e_est = sig2e_est,
            sig2b_est0 = sig2b0_est,
            sig2b_est1 = sig2b1_est,
            n_epochs = 0,
            time = end - start
          )
          }
      }
  }
  }
}

res_df <- bind_rows(res_list)
