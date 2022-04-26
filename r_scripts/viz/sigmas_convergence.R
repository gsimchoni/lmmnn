library(tidyverse)
library(patchwork)
library(extrafont)
font_import()
loadfonts(device="win") 

df <- read_csv("results/res_params_Z.csv")

df %>% count(experiment)

min_epochs <- df %>% count(experiment) %>% slice_min(n) %>% pull(n)

df_long <- df %>% pivot_longer(starts_with("sig2"), "sig2", values_to = "est") %>%
  mutate(epoch = epoch + 1) %>%
  bind_rows(
    tibble(
      epoch = 0, experiment = rep(0:9, each = 4), loss = 0, val_loss = 0,
      sig2 = rep(c("sig2b_est0", "sig2b_est1", "sig2b_est2", "sig2e_est"), 10),
      est = 1
    )
  )

mean_profile <- df_long %>% group_by(epoch, sig2) %>%
  summarise(est = mean(est)) %>%
  mutate(experiment = 0) %>%
  select(experiment, everything())

max_epochs <- 45 #25 for g(Z) = ZW
max_y <- 12

df_long %>%
  filter(epoch <= max_epochs, sig2 != "sig2e_est") %>%
  # mutate(experiment = experiment + 1, epoch = epoch + 1) %>%
  ggplot(aes(epoch, est, color = sig2, group = interaction(experiment, sig2))) +
  geom_line(alpha = 0.4) +
  geom_line(data = mean_profile %>% filter(epoch <= max_epochs, sig2 != "sig2e_est"), lwd = 1.2) +
  geom_hline(yintercept = 10.0, color = "blue", lty = 3, lwd = 1.2) +
  geom_hline(yintercept = 1.0, color = "green", lty = 3, lwd = 1.2) +
  geom_hline(yintercept = 0.1, color = "red", lty = 3, lwd = 1.2) +
  # scale_y_log10() +
  scale_y_continuous(breaks = 10^c(-1:1), limits = c(0, max_y)) +
  scale_color_discrete(labels = c(
    expression(paste(hat(sigma)[b[1]]^2)),
    expression(paste(hat(sigma)[b[2]]^2)),
    expression(paste(hat(sigma)[b[3]]^2)))) +
  labs(y = NULL, color = NULL, title = expression("g(Z) = Z")) +
  theme_bw() +
  theme(legend.position = "bottom",
        text = element_text(family = "Century", size = 16),
        plot.title = element_text(hjust = 0.5))
  
p1 <- df_long %>%
  filter(epoch <= max_epochs, sig2 != "sig2e_est") %>%
  # mutate(experiment = experiment + 1, epoch = epoch + 1) %>%
  ggplot(aes(epoch, est, color = sig2, group = interaction(experiment, sig2))) +
  geom_line(alpha = 0.4) +
  geom_line(data = mean_profile %>% filter(epoch <= max_epochs, sig2 != "sig2e_est"), lwd = 1.2) +
  geom_hline(yintercept = 10.0, color = "blue", lty = 3, lwd = 1.2) +
  geom_hline(yintercept = 1.0, color = "green", lty = 3, lwd = 1.2) +
  geom_hline(yintercept = 0.1, color = "red", lty = 3, lwd = 1.2) +
  # scale_y_log10() +
  scale_y_continuous(breaks = 10^c(-1:1), limits = c(0, max_y)) +
  scale_color_discrete(labels = c(
    expression(paste(hat(sigma)[b[1]]^2)),
    expression(paste(hat(sigma)[b[2]]^2)),
    expression(paste(hat(sigma)[b[3]]^2)))) +
  labs(y = NULL, color = NULL, title = expression("g(Z) = Z")) +
  theme_bw() +
  theme(legend.position = "bottom",
        text = element_text(family = "Century", size = 16),
        plot.title = element_text(hjust = 0.5))

df <- read_csv("results/res_params_gZ.csv")

df_long <- df %>% pivot_longer(starts_with("sig2"), "sig2", values_to = "est") %>%
  mutate(epoch = epoch + 1) %>%
  bind_rows(
    tibble(
      epoch = 0, experiment = rep(0:9, each = 4), loss = 0, val_loss = 0,
      sig2 = rep(c("sig2b_est0", "sig2b_est1", "sig2b_est2", "sig2e_est"), 10),
      est = 1
    )
  )

mean_profile <- df_long %>% group_by(epoch, sig2) %>%
  summarise(est = mean(est)) %>%
  mutate(experiment = 0) %>%
  select(experiment, everything())

max_epochs <- 25

p2 <- df_long %>%
  filter(epoch <= max_epochs, sig2 != "sig2e_est") %>%
  # mutate(experiment = experiment + 1, epoch = epoch + 1) %>%
  ggplot(aes(epoch, est, color = sig2, group = interaction(experiment, sig2))) +
  geom_line(alpha = 0.4) +
  geom_line(data = mean_profile %>% filter(epoch <= max_epochs, sig2 != "sig2e_est"), lwd = 1.2) +
  geom_hline(yintercept = 10.0, color = "blue", lty = 3, lwd = 1.2) +
  geom_hline(yintercept = 1.0, color = "green", lty = 3, lwd = 1.2) +
  geom_hline(yintercept = 0.1, color = "red", lty = 3, lwd = 1.2) +
  # scale_y_log10() +
  scale_y_continuous(breaks = 10^c(-1:1), limits = c(0, max_y), labels = NULL) +
  scale_color_discrete(labels = c(
    expression(paste(hat(sigma)[b[1]]^2)),
    expression(paste(hat(sigma)[b[2]]^2)),
    expression(paste(hat(sigma)[b[3]]^2)))) +
  labs(y = NULL, color = NULL, title = expression("g(Z) = ZW")) +
  theme_bw() +
  theme(legend.position = "bottom",
        text = element_text(family = "Century", size = 16),
        plot.title = element_text(hjust = 0.5))

p1 | p2
