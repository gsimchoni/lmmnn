library(tidyverse)
library(xtable)

res <- read_csv("res.csv")

mse_df <- res %>%
  group_by(q, sig2b, exp_type) %>%
  summarise(mse = mean(mse)) %>%
  pivot_wider(names_from = exp_type, values_from = mse) %>%
  select(ignore, ohe, embed, lme4, lmm)

xtable(mse_df, type = "latex")

sig2e_est_df <- res %>%
  filter(exp_type %in% c("lmm", "lme4")) %>%
  group_by(q, sig2b, exp_type) %>%
  summarise(avg_sig2e = mean(sig2e_est)) %>%
  pivot_wider(names_from = exp_type, values_from = avg_sig2e, names_prefix = "sig2e_") %>%
  ungroup()

sig2b_est_df <- res %>%
  filter(exp_type %in% c("lmm", "lme4")) %>%
  group_by(q, sig2b, exp_type) %>%
  summarise(avg_sig2b = mean(sig2b_est)) %>%
  pivot_wider(names_from = exp_type, values_from = avg_sig2b, names_prefix = "sig2b_") %>%
  ungroup()

sigmas_df <- bind_cols(sig2e_est_df, sig2b_est_df %>% select(-q, -sig2b)) %>%
  select(q, sig2b, ends_with("lme4"), ends_with("lmm"))

xtable(sigmas_df, type = "latex")

res %>%
  group_by(q, sig2b, exp_type) %>%
  summarise(n_epochs = mean(n_epochs)) %>%
  pivot_wider(names_from = exp_type, values_from = n_epochs)

res %>%
  group_by(q, sig2b, exp_type) %>%
  summarise(time = mean(time)) %>%
  pivot_wider(names_from = exp_type, values_from = time)
