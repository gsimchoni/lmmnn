library(tidyverse)
library(reticulate)
library(survival)

lmmnn_utils <- import("lmmnn.utils")

weibull_lambda <- 1.0
weibull_nu <- 2.0
cox <- TRUE # TRUE for coxph, FALSE for survreg
params_dict <- dict(n_fixed_effects = 10L, n_per_cat = 3, fixed_intercept = 1,
                    X_non_linear = TRUE, Z_non_linear = FALSE, Z_embed_dim_pct = 10,
                    weibull_lambda = weibull_lambda, weibull_nu = weibull_nu)
mode <- "survival"
sig2e <- 1.0
sig2b_list <- c(0.1, 1.0, 10.0)
q_list <- c(100L, 1000L, 10000L)
N <- 100000L
n_iter <- 5
p_censor_list <- c(1, 10, 30)

res_list <- list()
counter <- 0

for (sig2b in sig2b_list) {
  for (q in q_list) {
    for (p_censor in p_censor_list) {
      cat(glue::glue("N: {N}, sig2e: {sig2e}; sig2b: {sig2b}; q: {q}; p_censor: {p_censor}"), "\n")
      for (k in seq_len(n_iter)) {
        counter <- counter + 1
        cat(glue::glue("  iteration: {k}"), "\n")
        py_res <- lmmnn_utils$generate_data(
          mode = mode, qs = list(q), sig2e = sig2e, sig2bs = list(sig2b), N = N,
          rhos = list(), p_censor = p_censor, params = params_dict)
        X_train = py_res[[1]]
        X_test = py_res[[2]]
        y_train = py_res[[3]]
        y_test = py_res[[4]]
        
        df_train <- data.frame(y = y_train)
        df_train <- cbind(df_train, X_train)
        df_test <- data.frame(y = y_test)
        df_test <- cbind(df_test, X_test)
        
        surv <- survival::Surv(df_train$y, df_train$C0)
        if (cox) {
          form <- as.formula(str_c("surv ~ ", str_c(str_c(str_c("X", 0:9), collapse = " + "), " + frailty(z0)")))
        } else {
          form <- as.formula(str_c("surv ~ ", str_c("0 +", str_c(str_c("X", 0:9), collapse = " + "), " + frailty(z0)")))
        }
        start <- Sys.time()
        if (cox) {
          out <- coxph(form, df_train)
        } else {
          out <- survreg(form, df_train, dist = "weibull")
        }
        end <- Sys.time()
        
        sig2e_est <- ""
        sig2b_est <- out$history$`frailty(z0)`$theta
        
        if (cox) {
          y_pred <- predict(out, df_test, allow.new.levels = TRUE, type = "lp")
        } else {
          out_coef <- coef(out)
          out_coef[is.na(out_coef)] <- 0
          y_pred <- as.matrix(df_test[paste0("X", 0:9)]) %*% t(t(out_coef))
        }
        
        b_hat <- rep(0, q)
        b_hat[sort(unique(df_train$z0 + 1))] <- out$frail
        if (cox) {
          conc_data <- survConcordance.fit(Surv(df_test$y, df_test$C0), y_pred + b_hat[df_test[["z0"]] + 1])
        } else {
          conc_data <- survConcordance.fit(Surv(df_test$y, df_test$C0), -(y_pred + b_hat[df_test[["z0"]] + 1]))
        }
        
        conc <- conc_data[1] / (conc_data[1] + conc_data[2])
        res_list[[counter]] <- list(
          mode = mode,
          N = N,
          sig2e = sig2e,
          sig2b0 = sig2b,
          q0 = q,
          p_censor = p_censor,
          weibull_nu = weibull_nu,
          weibull_lambda = weibull_lambda,
          experiment = k,
          exp_type = ifelse(cox, "coxph", "survreg"),
          concordance = conc,
          sig2e_est = sig2e_est,
          sig2b_est0 = sig2b_est,
          weibull_nu_est = "",
          weibull_lambda_est = "",
          n_epochs = 0,
          time = difftime(end, start, units = "secs")[[1]]
        )
      }
    }
  }
}

res_df <- bind_rows(res_list)
