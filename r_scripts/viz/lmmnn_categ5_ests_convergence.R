library(tidyverse)
library(patchwork)
library(extrafont)
font_import()
loadfonts(device="win") 


# g(Z) = Z, left panel ----------------------------------------------------

df <- read_csv("data/data_for_viz/res_grads_n100K_q1000_Z_categ5_colab.csv")

df %>% count(experiment)

df_long <- df %>% pivot_longer(contains("grad"), "sig2", values_to = "grad") %>%
  mutate(epoch = epoch + 1) %>%
  filter(sig2 != "sig2e_grad") %>%
  bind_rows(
    tibble(
      epoch = 0, experiment = rep(0:4, each = 5), loss = 0, val_loss = 0,
      sig2 = rep(c("sig2b_grad0", "sig2b_grad1", "sig2b_grad2", "sig2b_grad3", "sig2b_grad4"), 5),
      est = 1
    )
  )

mean_profile <- df_long %>% group_by(epoch, sig2) %>%
  summarise(grad = mean(grad)) %>%
  mutate(experiment = 0) %>%
  select(experiment, everything())

max_epochs <- 40
min_y <- -5
max_y <- 5
q <- 1000

p2 <- df_long %>%
  filter(epoch <= max_epochs) %>%
  ggplot(aes(epoch, grad, color = sig2, group = interaction(experiment, sig2))) +
  geom_line(alpha = 0.4) +
  geom_line(data = mean_profile %>% filter(epoch <= max_epochs), lwd = 1.2) +
  geom_hline(yintercept = 0, color = "black", lty = 3, lwd = 1.2) +
  scale_y_continuous(breaks = c(-10, -5, 0), limits = c(-12, 2)) +
  # scale_y_continuous(limits = c(-10, breaks = c(-2, -1, 0, 1), minor_breaks = c(-2, -1, 0, 1), labels = c(-2, -1, 0, 1)) +
  scale_x_continuous(breaks = seq(0, max_epochs, 10)) +
  scale_color_discrete(labels = c(
    expression(paste(nabla, sigma[b[1]]^2)),
    expression(paste(nabla, sigma[b[2]]^2)),
    expression(paste(nabla, sigma[b[3]]^2)),
    expression(paste(nabla, sigma[b[4]]^2)),
    expression(paste(nabla, sigma[b[5]]^2)))) +
  labs(y = NULL, color = NULL, x = NULL, subtitle = "Gradients") +
  theme_bw() +
  theme(legend.position = "left",
        text = element_text(family = "Century", size = 14),
        plot.title = element_text(hjust = 0, size = 14),
        axis.text.x = element_blank(),
        plot.subtitle = element_text(hjust = 0, size = 12))


df_long <- df %>% pivot_longer(contains("est"), "sig2", values_to = "est") %>%
  mutate(epoch = epoch + 1) %>%
  bind_rows(
    tibble(
      epoch = 0, experiment = rep(0:4, each = 5), loss = 0, val_loss = 0,
      sig2 = rep(c("sig2b_est0", "sig2b_est1", "sig2b_est2", "sig2b_est3", "sig2b_est4"), 5),
      est = 1
    )
  )
mean_profile <- df_long %>% group_by(epoch, sig2) %>%
  summarise(est = mean(est)) %>%
  mutate(experiment = 0) %>%
  select(experiment, everything())

max_y <- 12

p1 <- df_long %>%
  filter(epoch <= max_epochs, sig2 != "sig2e_est") %>%
  ggplot(aes(epoch, est, color = sig2, group = interaction(experiment, sig2))) +
  geom_line(alpha = 0.4) +
  geom_line(data = mean_profile %>% filter(epoch <= max_epochs, sig2 != "sig2e_est"), lwd = 1.2) +
  geom_hline(yintercept = 1.0, color = "#F8766D", lty = 3, lwd = 1.2) +
  geom_hline(yintercept = 2.0, color = "#A3A500", lty = 3, lwd = 1.2) +
  geom_hline(yintercept = 3.0, color = "#00BF7D", lty = 3, lwd = 1.2) +
  geom_hline(yintercept = 4.0, color = "#00B0F6", lty = 3, lwd = 1.2) +
  geom_hline(yintercept = 5.0, color = "#E76BF3", lty = 3, lwd = 1.2) +
  scale_y_continuous(breaks = c(1:5, 7, 9, 11), labels = c(1:5, 7, 9, 11),
                     minor_breaks = 1:max_y, limits = c(0, max_y)) +
  scale_x_continuous(breaks = seq(0, max_epochs, 10)) +
  scale_color_discrete(labels = c(
    expression(paste(hat(sigma)[b[1]]^2)),
    expression(paste(hat(sigma)[b[2]]^2)),
    expression(paste(hat(sigma)[b[3]]^2)),
    expression(paste(hat(sigma)[b[4]]^2)),
    expression(paste(hat(sigma)[b[5]]^2)))) +
  labs(y = NULL, color = NULL, x = NULL,
       subtitle = "Estimates", title = "g(Z) = Z") +
  theme_bw() +
  theme(legend.position = "left",
        text = element_text(family = "Century", size = 14),
        plot.title = element_text(hjust = 0.5, size = 18),
        axis.text.x = element_blank(),
        plot.subtitle = element_text(hjust = 0, size = 12))


df_long <- df %>% pivot_longer(contains("loss"), "type", values_to = "loss")

mean_profile <- df_long %>% group_by(epoch, type) %>%
  summarise(loss = mean(loss)) %>%
  mutate(experiment = 0) %>%
  select(experiment, everything())

p3 <- df_long %>%
  filter(epoch <= max_epochs) %>%
  ggplot(aes(epoch, loss, color = type, group = interaction(experiment, type))) +
  geom_line(alpha = 0.4) +
  geom_line(data = mean_profile %>% filter(epoch <= max_epochs), lwd = 1.2) +
  scale_y_continuous(breaks = c(200, 400, 600), limits = c(0, 650)) +
  scale_x_continuous(breaks = seq(0, max_epochs, 10)) +
  scale_color_manual(labels = c("train", "valid"), values = c("#eb3434", "#3a34eb")) +
  labs(y = NULL, color = NULL, subtitle = "NLL Loss") +
  theme_bw() +
  theme(legend.position = "left",
        text = element_text(family = "Century", size = 14),
        plot.title = element_text(hjust = 0, size = 14),
        plot.subtitle = element_text(hjust = 0, size = 12))

p_Z <- p1 / p2 / p3 


# g(Z) = ZW, right panel --------------------------------------------------


df <- read_csv("data/data_for_viz/res_params_categ5_gZ.csv")

df %>% count(experiment)

df_long <- df %>% pivot_longer(contains("grad"), "sig2", values_to = "grad") %>%
  mutate(epoch = epoch + 1) %>%
  filter(sig2 != "sig2e_grad") %>%
  bind_rows(
    tibble(
      epoch = 0, experiment = rep(0:4, each = 5), loss = 0, val_loss = 0,
      sig2 = rep(c("sig2b_grad0", "sig2b_grad1", "sig2b_grad2", "sig2b_grad3", "sig2b_grad4"), 5),
      est = 1
    )
  )

mean_profile <- df_long %>% group_by(epoch, sig2) %>%
  summarise(grad = mean(grad)) %>%
  mutate(experiment = 0) %>%
  select(experiment, everything())

max_epochs <- 40
min_y <- -5
max_y <- 5
q <- 1000

p2 <- df_long %>%
  filter(epoch <= max_epochs) %>%
  ggplot(aes(epoch, grad, color = sig2, group = interaction(experiment, sig2))) +
  geom_line(alpha = 0.4) +
  geom_line(data = mean_profile %>% filter(epoch <= max_epochs), lwd = 1.2) +
  geom_hline(yintercept = 0, color = "black", lty = 3, lwd = 1.2) +
  scale_y_continuous(breaks = c(-10, -5, 0), limits = c(-12, 2)) +
  scale_x_continuous(breaks = seq(0, max_epochs, 10)) +
  labs(y = NULL, color = NULL, x = NULL, subtitle = "") +
  guides(color = "none") +
  theme_bw() +
  theme(legend.position = "left",
        text = element_text(family = "Century", size = 14),
        plot.title = element_text(hjust = 0, size = 14),
        axis.text.x = element_blank(),
        plot.subtitle = element_text(hjust = 0, size = 12))


df_long <- df %>% pivot_longer(contains("est"), "sig2", values_to = "est") %>%
  mutate(epoch = epoch + 1) %>%
  bind_rows(
    tibble(
      epoch = 0, experiment = rep(0:4, each = 5), loss = 0, val_loss = 0,
      sig2 = rep(c("sig2b_est0", "sig2b_est1", "sig2b_est2", "sig2b_est3", "sig2b_est4"), 5),
      est = 1
    )
  )
mean_profile <- df_long %>% group_by(epoch, sig2) %>%
  summarise(est = mean(est)) %>%
  mutate(experiment = 0) %>%
  select(experiment, everything())

max_y <- 12

p1 <- df_long %>%
  filter(epoch <= max_epochs, sig2 != "sig2e_est") %>%
  ggplot(aes(epoch, est, color = sig2, group = interaction(experiment, sig2))) +
  geom_line(alpha = 0.4) +
  geom_line(data = mean_profile %>% filter(epoch <= max_epochs, sig2 != "sig2e_est"), lwd = 1.2) +
  geom_hline(yintercept = 1.0, color = "#F8766D", lty = 3, lwd = 1.2) +
  geom_hline(yintercept = 2.0, color = "#A3A500", lty = 3, lwd = 1.2) +
  geom_hline(yintercept = 3.0, color = "#00BF7D", lty = 3, lwd = 1.2) +
  geom_hline(yintercept = 4.0, color = "#00B0F6", lty = 3, lwd = 1.2) +
  geom_hline(yintercept = 5.0, color = "#E76BF3", lty = 3, lwd = 1.2) +
  scale_y_continuous(breaks = c(1:5, 7, 9, 11), labels = c(1:5, 7, 9, 11), minor_breaks = 1:max_y, limits = c(0, max_y)) +
  scale_x_continuous(breaks = seq(0, max_epochs, 10)) +
  labs(y = NULL, color = NULL, x = NULL,
       subtitle = NULL, title = "g(Z) = ZW") +
  guides(color = "none") +
  theme_bw() +
  theme(legend.position = "left",
        text = element_text(family = "Century", size = 14),
        plot.title = element_text(hjust = 0.5, size = 18),
        axis.text.x = element_blank(),
        plot.subtitle = element_text(hjust = 0, size = 12))


df_long <- df %>% pivot_longer(contains("loss"), "type", values_to = "loss")

mean_profile <- df_long %>% group_by(epoch, type) %>%
  summarise(loss = mean(loss)) %>%
  mutate(experiment = 0) %>%
  select(experiment, everything())

p3 <- df_long %>%
  filter(epoch <= max_epochs) %>%
  ggplot(aes(epoch, loss, color = type, group = interaction(experiment, type))) +
  geom_line(alpha = 0.4) +
  geom_line(data = mean_profile %>% filter(epoch <= max_epochs), lwd = 1.2) +
  scale_y_continuous(breaks = c(200, 400, 600), limits = c(0, 650)) +
  scale_x_continuous(breaks = seq(0, max_epochs, 10)) +
  scale_color_manual(labels = c("train", "valid"), values = c("#eb3434", "#3a34eb")) +
  labs(y = NULL, color = NULL, subtitle = "") +
  guides(color = "none") +
  theme_bw() +
  theme(legend.position = "left",
        text = element_text(family = "Century", size = 14),
        plot.title = element_text(hjust = 0, size = 14),
        plot.subtitle = element_text(hjust = 0, size = 12))

p_ZW <- p1 / p2 / p3 


# Unite two panels --------------------------------------------------------


p_Z | p_ZW
ggsave("images/lmmnn_categ5_ests_convergence.png", width = 8, height = 7, device="png", dpi=700)
