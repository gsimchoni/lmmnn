library(tidyverse)
library(reticulate)
library(gstat)
library(sp)

lmmnn_utils <- import("lmmnn.utils")
params_dict <- dict(n_fixed_effects = 10L, n_per_cat = 3, fixed_intercept = 1,
                    X_non_linear = TRUE, Z_non_linear = FALSE, Z_embed_dim_pct = 10)
mode <- "spatial"
sig2e <- 1.0
sig2b0_list <- c(0.1, 1.0)
sig2b1_list <- c(0.1, 1.0, 10.0)
q_list <- c(100L, 1000L)
N <- 10000L
n_iter <- 5

res_list <- list()
counter <- 0

for (sig2b0 in sig2b0_list) {
  for (sig2b1 in sig2b1_list) {
    for (q in q_list) {
      cat(glue::glue("N: {N}, sig2e: {sig2e}; sig2b0: {sig2b0}; sig2b1: {sig2b1}; q: {q}"), "\n")
      for (k in seq_len(n_iter)) {
        counter <- counter + 1
        cat(glue::glue("  iteration: {k}"), "\n")
        py_res <- lmmnn_utils$generate_data(
          mode = mode, qs = list(q), sig2e = sig2e, sig2bs = list(sig2b0, sig2b1),
          N = N, rhos = list(), p_censor = list(), params = params_dict)
        X_train <- py_res[[1]]
        X_test <- py_res[[2]]
        y_train <- py_res[[3]]
        y_test <- py_res[[4]]
        
        df_train <- data.frame(y = y_train)
        df_train <- cbind(df_train, X_train)
        df_test <- data.frame(y = y_test)
        df_test <- cbind(df_test, X_test)
        
        # df_train = df_train[which(!duplicated(df_train[c("D1", "D2")])), ]
        df_train$D1 <- df_train$D1 + rnorm(nrow(df_train), sd = 0.001)
        df_train$D2 <- df_train$D2 + rnorm(nrow(df_train), sd = 0.001)
        coordinates(df_train) <-  ~ D1 + D2
        # df_test = df_test[which(!duplicated(df_test[c("D1", "D2")])), ]
        df_test$D1 <- df_test$D1 + rnorm(nrow(df_test), sd = 0.001)
        df_test$D2 <- df_test$D2 + rnorm(nrow(df_test), sd = 0.001)
        coordinates(df_test) <-  ~ D1 + D2
        
        form <- as.formula(str_c("y ~ ", str_c(str_c(str_c("X", 0:9), collapse = " + "))))
        start <- Sys.time()
        try({
          vgm_obj <- variogram(form, df_train)
          fit_obj <- fit.variogram(vgm_obj, model = vgm("Gau"))
          y_pred <- krige(form, df_train, df_test, model = fit_obj)$var1.pred
          end <- Sys.time()
          
          sig2e_est <- fit_obj$psill[1]
          sig2b0_est <- fit_obj$psill[2]
          sig2b1_est <- (fit_obj$range[2]^2)/2
          
          mse <- mean((y_test - y_pred)^2)
          res_list[[counter]] <- list(
            mode = mode,
            N = N,
            sig2e = sig2e,
            sig2b0 = sig2b0,
            sig2b1 = sig2b1,
            q = q,
            experiment = k,
            exp_type = "krig",
            mse = mse,
            sig2e_est = sig2e_est,
            sig2b0_est = sig2b0_est,
            sig2b1_est = sig2b1_est,
            n_epochs = 0,
            time = end - start
          )
        })
        
      }
    }
  }
}

res_df <- bind_rows(res_list)
