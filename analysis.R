library(tidyverse)

res <- read_csv("../lmmnn/res.csv")

res %>%
  group_by(q, sig2b, deep, exp_type) %>%
  summarise(mse = mean(mse)) %>%
  pivot_wider(names_from = exp_type, values_from = mse)

res %>%
  filter(exp_type == "lmm") %>%
  group_by(q, sig2b, deep) %>%
  summarise(avg_sig2e = mean(sig2e_est), avg_sig2b = mean(sig2b_est))

res %>%
  group_by(q, sig2b, deep, exp_type) %>%
  summarise(n_epochs = mean(n_epochs)) %>%
  pivot_wider(names_from = exp_type, values_from = n_epochs)

res %>%
  group_by(q, sig2b, deep, exp_type) %>%
  summarise(time = mean(time)) %>%
  pivot_wider(names_from = exp_type, values_from = time)
