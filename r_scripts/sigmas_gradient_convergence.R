library(tidyverse)
library(patchwork)
library(extrafont)
#font_import()
loadfonts(device="win") 

df <- read_csv("~/res_grads_n10K_q1000_sig2b10.csv")
df %>% count(experiment)

min_epochs <- df %>% count(experiment) %>% slice_min(n) %>% pull(n)

df_long <- df %>% pivot_longer(contains("grad"), "sig2", values_to = "grad") %>%
  mutate(epoch = epoch + 1) #%>%
  # bind_rows(
  #   tibble(
  #     epoch = 0, experiment = rep(0:4, each = 2), loss = 0, val_loss = 0,
  #     sig2 = rep(c("sig2e_grad", "sig2b_grad0"), 5),
  #     grad = 1
  #   )
  # )

mean_profile <- df_long %>% group_by(epoch, sig2) %>%
  summarise(grad = mean(grad)) %>%
  mutate(experiment = 0) %>%
  select(experiment, everything())

max_epochs <- 45 #25 for g(Z) = ZW
max_y <- 7
q <- 1000
sig2b <- 10

df_long %>%
  filter(epoch <= max_epochs) %>%
  # mutate(experiment = experiment + 1, epoch = epoch + 1) %>%
  ggplot(aes(epoch, grad, color = sig2, group = interaction(experiment, sig2))) +
  geom_line(alpha = 0.4) +
  geom_line(data = mean_profile %>% filter(epoch <= max_epochs), lwd = 1.2) +
  # geom_hline(yintercept = 10.0, color = "blue", lty = 3, lwd = 1.2) +
  # geom_hline(yintercept = 1.0, color = "green", lty = 3, lwd = 1.2) +
  geom_hline(yintercept = 0, color = "black", lty = 3, lwd = 1.2) +
  # scale_y_log10() +
  scale_y_continuous(breaks = 2*c(-3:3), limits = c(-7, max_y)) +
  scale_color_discrete(labels = c(
    expression(paste("  ", nabla, sigma[e]^2)),
    expression(paste("  ", nabla, sigma[b]^2)))) +
  labs(y = NULL, color = NULL, title = bquote(paste("n = 10,000; Single categorical w/ q = ", .(q), ", ",
    sigma[b]^2, " = ", .(sig2b), "; g(Z) = Z"))) +
  theme_bw() +
  theme(legend.position = "bottom",
        text = element_text(family = "Century", size = 16),
        plot.title = element_text(hjust = 0.5, size = 12))


### multi categ

df <- read_csv("~/res_sigmas_grads_n100K_categ3_q1000_NL.csv")

df_long <- df %>% pivot_longer(contains("grad"), "sig2", values_to = "grad") %>%
  mutate(epoch = epoch + 1) %>%
  filter(sig2 != "sig2e_grad") %>%
  bind_rows(
    tibble(
      epoch = 0, experiment = rep(0:4, each = 3), loss = 0, val_loss = 0,
      sig2 = rep(c("sig2b_grad0", "sig2b_grad1", "sig2b_grad2"), 5)
    )
  )

mean_profile <- df_long %>% group_by(epoch, sig2) %>%
  summarise(grad = mean(grad)) %>%
  mutate(experiment = 0) %>%
  select(experiment, everything())

max_epochs <- 30
min_y <- -10
max_y <- 5
q <- 1000
sig2b0 <- 0.1
sig2b1 <- 1.0
sig2b2 <- 10.0

p2 <- df_long %>%
  filter(epoch <= max_epochs) %>%
  ggplot(aes(epoch, grad, color = sig2, group = interaction(experiment, sig2))) +
  geom_line(alpha = 0.4) +
  geom_line(data = mean_profile %>% filter(epoch <= max_epochs), lwd = 1.2) +
  geom_hline(yintercept = 0, color = "black", lty = 3, lwd = 1.2) +
  scale_y_continuous(breaks = c(-10, -5, 0, 5, 10), limits = c(min_y, max_y)) +
  scale_x_continuous(breaks = seq(0, 30, 5)) +
  scale_color_discrete(labels = c(
    expression(paste("  ", nabla, sigma[b[1]]^2)),
    expression(paste("  ", nabla, sigma[b[2]]^2)),
    expression(paste("  ", nabla, sigma[b[3]]^2)))) +
  labs(y = NULL, color = NULL, x = NULL, title = "Gradients") +
  theme_bw() +
  theme(legend.position = "left",
        text = element_text(family = "Century", size = 14),
        plot.title = element_text(hjust = 0, size = 14),
        axis.text.x = element_blank())


df_long <- df %>% pivot_longer(contains("est"), "sig2", values_to = "est") %>%
  mutate(epoch = epoch + 1) %>%
  bind_rows(
    tibble(
      epoch = 0, experiment = rep(0:4, each = 4), loss = 0, val_loss = 0,
      sig2 = rep(c("sig2b_est0", "sig2b_est1", "sig2b_est2", "sig2e_est"), 5),
      est = 1
    )
  )
mean_profile <- df_long %>% group_by(epoch, sig2) %>%
  summarise(est = mean(est)) %>%
  mutate(experiment = 0) %>%
  select(experiment, everything())

max_epochs <- 30
max_y <- 15

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
  scale_x_continuous(breaks = seq(0, 30, 5)) +
  scale_color_discrete(labels = c(
    expression(paste(hat(sigma)[b[1]]^2)),
    expression(paste(hat(sigma)[b[2]]^2)),
    expression(paste(hat(sigma)[b[3]]^2)))) +
  labs(y = NULL, color = NULL, x = NULL,
       title = "Estimates") +
  theme_bw() +
  theme(legend.position = "left",
        text = element_text(family = "Century", size = 14),
        plot.title = element_text(hjust = 0, size = 14),
        axis.text.x = element_blank())


df_long <- df %>% pivot_longer(contains("loss"), "type", values_to = "loss")

mean_profile <- df_long %>% group_by(epoch, type) %>%
  summarise(loss = mean(loss)) %>%
  mutate(experiment = 0) %>%
  select(experiment, everything())

p3 <- df_long %>%
  filter(epoch <= max_epochs) %>%
  # mutate(experiment = experiment + 1, epoch = epoch + 1) %>%
  ggplot(aes(epoch, loss, color = type, group = interaction(experiment, type))) +
  geom_line(alpha = 0.4) +
  geom_line(data = mean_profile %>% filter(epoch <= max_epochs), lwd = 1.2) +
  geom_hline(yintercept = 0, color = "black", lty = 3, lwd = 1.2) +
  scale_y_log10() +
  # scale_y_continuous(breaks = 10^c(-1:1), limits = c(0, max_y)) +
  scale_x_continuous(breaks = seq(0, 30, 5)) +
  scale_color_manual(labels = c("train", "val"), values = c("yellow", "purple")) +
  labs(y = NULL, color = NULL, title = "Loss") +
  theme_bw() +
  theme(legend.position = "left",
        text = element_text(family = "Century", size = 14),
        plot.title = element_text(hjust = 0, size = 14))

sig2b0 <- 0.1
sig2b1 <- 1
sig2b2 <- 10
q <- 1000
p <- p1 / p2 / p3 
p + plot_annotation(title = "Three categoricals, g(Z) = ZW",
                    caption = expression(paste("n = 100,000; ", " q = ", 1000, ", ",
                                            sigma[b[1]]^2, " = ", 0.1, ", ", sigma[b[2]]^2, " = ", 1,
                                            ", ", sigma[b[3]]^2, " = ", 10)),
                    theme = theme(text = element_text(family = "Century", size = 16),
                                  plot.title = element_text(hjust = 0.5)))

####

df <- read_csv("~/res_sigmas_grads_n100K_categ3_q1000.csv")

df_long <- df %>% pivot_longer(contains("grad"), "sig2", values_to = "grad") %>%
  mutate(epoch = epoch + 1) %>%
  filter(sig2 != "sig2e_grad") %>%
  bind_rows(
    tibble(
      epoch = 0, experiment = rep(0:4, each = 3), loss = 0, val_loss = 0,
      sig2 = rep(c("sig2b_grad0", "sig2b_grad1", "sig2b_grad2"), 5)
    )
  )

mean_profile <- df_long %>% group_by(epoch, sig2) %>%
  summarise(grad = mean(grad)) %>%
  mutate(experiment = 0) %>%
  select(experiment, everything())

max_epochs <- 30
min_y <- -15
max_y <- 5
q <- 1000
sig2b0 <- 0.1
sig2b1 <- 1.0
sig2b2 <- 10.0

p2 <- df_long %>%
  filter(epoch <= max_epochs) %>%
  ggplot(aes(epoch, grad, color = sig2, group = interaction(experiment, sig2))) +
  geom_line(alpha = 0.4) +
  geom_line(data = mean_profile %>% filter(epoch <= max_epochs), lwd = 1.2) +
  geom_hline(yintercept = 0, color = "black", lty = 3, lwd = 1.2) +
  scale_y_continuous(breaks = c(-15, -10, -5, 0, 5, 10), limits = c(min_y, max_y)) +
  scale_x_continuous(breaks = seq(0, 30, 5)) +
  scale_color_discrete(labels = c(
    expression(paste("  ", nabla, sigma[b[1]]^2)),
    expression(paste("  ", nabla, sigma[b[2]]^2)),
    expression(paste("  ", nabla, sigma[b[3]]^2)))) +
  labs(y = NULL, color = NULL, x = NULL, title = "Gradients") +
  theme_bw() +
  theme(legend.position = "left",
        text = element_text(family = "Century", size = 14),
        plot.title = element_text(hjust = 0, size = 14),
        axis.text.x = element_blank())


df_long <- df %>% pivot_longer(contains("est"), "sig2", values_to = "est") %>%
  mutate(epoch = epoch + 1) %>%
  bind_rows(
    tibble(
      epoch = 0, experiment = rep(0:4, each = 4), loss = 0, val_loss = 0,
      sig2 = rep(c("sig2b_est0", "sig2b_est1", "sig2b_est2", "sig2e_est"), 5),
      est = 1
    )
  )
mean_profile <- df_long %>% group_by(epoch, sig2) %>%
  summarise(est = mean(est)) %>%
  mutate(experiment = 0) %>%
  select(experiment, everything())

max_epochs <- 30
max_y <- 12

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
  scale_x_continuous(breaks = seq(0, 30, 5)) +
  scale_color_discrete(labels = c(
    expression(paste(hat(sigma)[b[1]]^2)),
    expression(paste(hat(sigma)[b[2]]^2)),
    expression(paste(hat(sigma)[b[3]]^2)))) +
  labs(y = NULL, color = NULL, x = NULL,
       title = "Estimates") +
  theme_bw() +
  theme(legend.position = "left",
        text = element_text(family = "Century", size = 14),
        plot.title = element_text(hjust = 0, size = 14),
        axis.text.x = element_blank())


df_long <- df %>% pivot_longer(contains("loss"), "type", values_to = "loss")

mean_profile <- df_long %>% group_by(epoch, type) %>%
  summarise(loss = mean(loss)) %>%
  mutate(experiment = 0) %>%
  select(experiment, everything())

p3 <- df_long %>%
  filter(epoch <= max_epochs) %>%
  # mutate(experiment = experiment + 1, epoch = epoch + 1) %>%
  ggplot(aes(epoch, loss, color = type, group = interaction(experiment, type))) +
  geom_line(alpha = 0.4) +
  geom_line(data = mean_profile %>% filter(epoch <= max_epochs), lwd = 1.2) +
  geom_hline(yintercept = 0, color = "black", lty = 3, lwd = 1.2) +
  scale_y_log10() +
  # scale_y_continuous(breaks = 10^c(-1:1), limits = c(0, max_y)) +
  scale_x_continuous(breaks = seq(0, 30, 5)) +
  scale_color_manual(labels = c("train", "val"), values = c("yellow", "purple")) +
  labs(y = NULL, color = NULL, title = "Loss") +
  theme_bw() +
  theme(legend.position = "left",
        text = element_text(family = "Century", size = 14),
        plot.title = element_text(hjust = 0, size = 14))

sig2b0 <- 0.1
sig2b1 <- 1
sig2b2 <- 10
q <- 1000
p <- p1 / p2 / p3 
p + plot_annotation(title = "Three categoricals, g(Z) = Z",
                    caption = expression(paste("n = 100,000; ", " q = ", 1000, ", ",
                                               sigma[b[1]]^2, " = ", 0.1, ", ", sigma[b[2]]^2, " = ", 1,
                                               ", ", sigma[b[3]]^2, " = ", 10)),
                    theme = theme(text = element_text(family = "Century", size = 16),
                                  plot.title = element_text(hjust = 0.5)))

############

df <- read_csv("~/res_grads_n10K_q1000_sig2b10.csv")

df %>% count(experiment)

df_long <- df %>% pivot_longer(contains("grad"), "sig2", values_to = "grad") %>%
  mutate(epoch = epoch + 1) %>%
  # filter(sig2 != "sig2e_grad") %>%
  bind_rows(
    tibble(
      epoch = 0, experiment = rep(0:4, each = 2), loss = 0, val_loss = 0,
      sig2 = rep(c("sig2e_grad", "sig2b_grad0"), 5),
      est = 1
    )
  )

mean_profile <- df_long %>% group_by(epoch, sig2) %>%
  summarise(grad = mean(grad)) %>%
  mutate(experiment = 0) %>%
  select(experiment, everything())

max_epochs <- 70
min_y <- -15
max_y <- 5
q <- 1000

p2 <- df_long %>%
  filter(epoch <= max_epochs) %>%
  ggplot(aes(epoch, grad, color = sig2, group = interaction(experiment, sig2))) +
  geom_line(alpha = 0.4) +
  geom_line(data = mean_profile %>% filter(epoch <= max_epochs), lwd = 1.2) +
  geom_hline(yintercept = 0, color = "black", lty = 3, lwd = 1.2) +
  scale_y_continuous(breaks = c(-15, -10, -5, 0, 5, 10), limits = c(min_y, max_y)) +
  scale_x_continuous(breaks = seq(0, max_epochs, 10)) +
  scale_color_manual(values = c("#F8766D", "#00BA38"), labels = c(
    expression(paste("  ", nabla, sigma[b]^2)),
    expression(paste("  ", nabla, sigma[e]^2)))) +
  labs(y = NULL, color = NULL, x = NULL, title = "Gradients") +
  theme_bw() +
  theme(legend.position = "left",
        text = element_text(family = "Century", size = 14),
        plot.title = element_text(hjust = 0, size = 14),
        axis.text.x = element_blank())


df_long <- df %>% pivot_longer(contains("est"), "sig2", values_to = "est") %>%
  mutate(epoch = epoch + 1) %>%
  bind_rows(
    tibble(
      epoch = 0, experiment = rep(0:4, each = 2), loss = 0, val_loss = 0,
      sig2 = rep(c("sig2e_est", "sig2b_est0"), 5),
      est = 1
    )
  )
mean_profile <- df_long %>% group_by(epoch, sig2) %>%
  summarise(est = mean(est)) %>%
  mutate(experiment = 0) %>%
  select(experiment, everything())

# max_epochs <- 30
max_y <- 12

p1 <- df_long %>%
  filter(epoch <= max_epochs) %>%
  # mutate(experiment = experiment + 1, epoch = epoch + 1) %>%
  ggplot(aes(epoch, est, color = sig2, group = interaction(experiment, sig2))) +
  geom_line(alpha = 0.4) +
  geom_line(data = mean_profile %>% filter(epoch <= max_epochs), lwd = 1.2) +
  # geom_hline(yintercept = 10.0, color = "blue", lty = 3, lwd = 1.2) +
  geom_hline(yintercept = 1.0, color = "#00BA38", lty = 3, lwd = 1.2) +
  geom_hline(yintercept = 10.0, color = "#F8766D", lty = 3, lwd = 1.2) +
  # scale_y_log10() +
  scale_y_continuous(breaks = c(1, 10), limits = c(0, max_y)) +
  scale_x_continuous(breaks = seq(0, max_epochs, 10)) +
  scale_color_manual(values =  c("#F8766D", "#00BA38"), labels = c(
    expression(paste(hat(sigma)[b]^2)),
    expression(paste(hat(sigma)[e]^2)))) +
  labs(y = NULL, color = NULL, x = NULL,
       title = "Estimates") +
  theme_bw() +
  theme(legend.position = "left",
        text = element_text(family = "Century", size = 14),
        plot.title = element_text(hjust = 0, size = 14),
        axis.text.x = element_blank())


df_long <- df %>% pivot_longer(contains("loss"), "type", values_to = "loss")

mean_profile <- df_long %>% group_by(epoch, type) %>%
  summarise(loss = mean(loss)) %>%
  mutate(experiment = 0) %>%
  select(experiment, everything())

p3 <- df_long %>%
  filter(epoch <= max_epochs) %>%
  # mutate(experiment = experiment + 1, epoch = epoch + 1) %>%
  ggplot(aes(epoch, loss, color = type, group = interaction(experiment, type))) +
  geom_line(alpha = 0.4) +
  geom_line(data = mean_profile %>% filter(epoch <= max_epochs), lwd = 1.2) +
  geom_hline(yintercept = 0, color = "black", lty = 3, lwd = 1.2) +
  scale_y_log10() +
  # scale_y_continuous(breaks = 10^c(-1:1), limits = c(0, max_y)) +
  scale_x_continuous(breaks = seq(0, max_epochs, 10)) +
  scale_color_manual(labels = c("train", "val"), values = c("yellow", "purple")) +
  labs(y = NULL, color = NULL, title = "Loss") +
  theme_bw() +
  theme(legend.position = "left",
        text = element_text(family = "Century", size = 14),
        plot.title = element_text(hjust = 0, size = 14))

p <- p1 / p2 / p3 
p + plot_annotation(title = "Single categorical, g(Z) = Z",
                    caption = expression(paste("n = 10,000; ", " q = ", 1000, ", ",
                                               sigma[b]^2, " = ", 10, ", ")),
                    theme = theme(text = element_text(family = "Century", size = 16),
                                  plot.title = element_text(hjust = 0.5)))

###


df <- read_csv("~/res_grads_n100K_q1000_sig2b10_NL_cd.csv")
df <- df %>% slice(248:n())
df %>% count(experiment)

df_long <- df %>% pivot_longer(contains("grad"), "sig2", values_to = "grad") %>%
  mutate(epoch = epoch + 1) %>%
  # filter(sig2 != "sig2e_grad") %>%
  bind_rows(
    tibble(
      epoch = 0, experiment = rep(0:4, each = 2), loss = 0, val_loss = 0,
      sig2 = rep(c("sig2e_grad", "sig2b_grad0"), 5),
      est = 1
    )
  )

mean_profile <- df_long %>% group_by(epoch, sig2) %>%
  summarise(grad = mean(grad)) %>%
  mutate(experiment = 0) %>%
  select(experiment, everything())

max_epochs <- 70
min_y <- -15
max_y <- 5
q <- 1000

p2 <- df_long %>%
  filter(epoch <= max_epochs) %>%
  ggplot(aes(epoch, grad, color = sig2, group = interaction(experiment, sig2))) +
  geom_line(alpha = 0.4) +
  geom_line(data = mean_profile %>% filter(epoch <= max_epochs), lwd = 1.2) +
  geom_hline(yintercept = 0, color = "black", lty = 3, lwd = 1.2) +
  scale_y_continuous(breaks = c(-15, -10, -5, 0, 5, 10), limits = c(min_y, max_y)) +
  scale_x_continuous(breaks = seq(0, max_epochs, 10)) +
  scale_color_manual(values = c("#F8766D", "#00BA38"), labels = c(
    expression(paste("  ", nabla, sigma[b]^2)),
    expression(paste("  ", nabla, sigma[e]^2)))) +
  labs(y = NULL, color = NULL, x = NULL, title = "Gradients") +
  theme_bw() +
  theme(legend.position = "left",
        text = element_text(family = "Century", size = 14),
        plot.title = element_text(hjust = 0, size = 14),
        axis.text.x = element_blank())


df_long <- df %>% pivot_longer(contains("est"), "sig2", values_to = "est") %>%
  mutate(epoch = epoch + 1) %>%
  bind_rows(
    tibble(
      epoch = 0, experiment = rep(0:4, each = 2), loss = 0, val_loss = 0,
      sig2 = rep(c("sig2e_est", "sig2b_est0"), 5),
      est = 1
    )
  )
mean_profile <- df_long %>% group_by(epoch, sig2) %>%
  summarise(est = mean(est)) %>%
  mutate(experiment = 0) %>%
  select(experiment, everything())

# max_epochs <- 30
max_y <- 12

p1 <- df_long %>%
  filter(epoch <= max_epochs) %>%
  # mutate(experiment = experiment + 1, epoch = epoch + 1) %>%
  ggplot(aes(epoch, est, color = sig2, group = interaction(experiment, sig2))) +
  geom_line(alpha = 0.4) +
  geom_line(data = mean_profile %>% filter(epoch <= max_epochs), lwd = 1.2) +
  # geom_hline(yintercept = 10.0, color = "blue", lty = 3, lwd = 1.2) +
  geom_hline(yintercept = 1.0, color = "#00BA38", lty = 3, lwd = 1.2) +
  geom_hline(yintercept = 10.0, color = "#F8766D", lty = 3, lwd = 1.2) +
  # scale_y_log10() +
  scale_y_continuous(breaks = c(1, 10), limits = c(0, max_y)) +
  scale_x_continuous(breaks = seq(0, max_epochs, 10)) +
  scale_color_manual(values =  c("#F8766D", "#00BA38"), labels = c(
    expression(paste(hat(sigma)[b]^2)),
    expression(paste(hat(sigma)[e]^2)))) +
  labs(y = NULL, color = NULL, x = NULL,
       title = "Estimates") +
  theme_bw() +
  theme(legend.position = "left",
        text = element_text(family = "Century", size = 14),
        plot.title = element_text(hjust = 0, size = 14),
        axis.text.x = element_blank())


df_long <- df %>% pivot_longer(contains("loss"), "type", values_to = "loss")

mean_profile <- df_long %>% group_by(epoch, type) %>%
  summarise(loss = mean(loss)) %>%
  mutate(experiment = 0) %>%
  select(experiment, everything())

p3 <- df_long %>%
  filter(epoch <= max_epochs) %>%
  # mutate(experiment = experiment + 1, epoch = epoch + 1) %>%
  ggplot(aes(epoch, loss, color = type, group = interaction(experiment, type))) +
  geom_line(alpha = 0.4) +
  geom_line(data = mean_profile %>% filter(epoch <= max_epochs), lwd = 1.2) +
  geom_hline(yintercept = 0, color = "black", lty = 3, lwd = 1.2) +
  scale_y_log10() +
  # scale_y_continuous(breaks = 10^c(-1:1), limits = c(0, max_y)) +
  scale_x_continuous(breaks = seq(0, max_epochs, 10)) +
  scale_color_manual(labels = c("train", "val"), values = c("yellow", "purple")) +
  labs(y = NULL, color = NULL, title = "Loss") +
  theme_bw() +
  theme(legend.position = "left",
        text = element_text(family = "Century", size = 14),
        plot.title = element_text(hjust = 0, size = 14))

p <- p1 / p2 / p3 
p + plot_annotation(title = "Single categorical, g(Z) = ZW",
                    caption = expression(paste("n = 100,000; ", " q = ", 1000, ", ",
                                               sigma[b]^2, " = ", 10, ", ")),
                    theme = theme(text = element_text(family = "Century", size = 16),
                                  plot.title = element_text(hjust = 0.5)))

####################

df <- read_csv("~/res_grads_n100K_q1000_Z_categ5_colab.csv")
# df <- df %>% slice(248:n())
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
  # scale_y_continuous(breaks = c(-15, -10, -5, 0, 5, 10), limits = c(min_y, max_y)) +
  scale_x_continuous(breaks = seq(0, max_epochs, 10)) +
  scale_color_discrete(labels = c(
    expression(paste("  ", nabla, sigma[b[1]]^2)),
    expression(paste("  ", nabla, sigma[b[2]]^2)),
    expression(paste("  ", nabla, sigma[b[3]]^2)),
    expression(paste("  ", nabla, sigma[b[4]]^2)),
    expression(paste("  ", nabla, sigma[b[5]]^2)))) +
  labs(y = NULL, color = NULL, x = NULL, title = "Gradients") +
  theme_bw() +
  theme(legend.position = "left",
        text = element_text(family = "Century", size = 14),
        plot.title = element_text(hjust = 0, size = 14),
        axis.text.x = element_blank())


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

# max_epochs <- 30
max_y <- 12

p1 <- df_long %>%
  filter(epoch <= max_epochs, sig2 != "sig2e_est") %>%
  # mutate(experiment = experiment + 1, epoch = epoch + 1) %>%
  ggplot(aes(epoch, est, color = sig2, group = interaction(experiment, sig2))) +
  geom_line(alpha = 0.4) +
  geom_line(data = mean_profile %>% filter(epoch <= max_epochs, sig2 != "sig2e_est"), lwd = 1.2) +
  # geom_hline(yintercept = 10.0, color = "blue", lty = 3, lwd = 1.2) +
  geom_hline(yintercept = 1.0, color = "#F8766D", lty = 3, lwd = 1.2) +
  geom_hline(yintercept = 2.0, color = "#A3A500", lty = 3, lwd = 1.2) +
  geom_hline(yintercept = 3.0, color = "#00BF7D", lty = 3, lwd = 1.2) +
  geom_hline(yintercept = 4.0, color = "#00B0F6", lty = 3, lwd = 1.2) +
  geom_hline(yintercept = 5.0, color = "#E76BF3", lty = 3, lwd = 1.2) +
  # scale_y_log10() +
  scale_y_continuous(breaks = c(-1, 6)) +
  scale_x_continuous(breaks = seq(0, max_epochs, 10)) +
  scale_color_discrete(labels = c(
    expression(paste("  ", nabla, sigma[b[1]]^2)),
    expression(paste("  ", nabla, sigma[b[2]]^2)),
    expression(paste("  ", nabla, sigma[b[3]]^2)),
    expression(paste("  ", nabla, sigma[b[4]]^2)),
    expression(paste("  ", nabla, sigma[b[5]]^2)))) +
  labs(y = NULL, color = NULL, x = NULL,
       title = "Estimates") +
  theme_bw() +
  theme(legend.position = "left",
        text = element_text(family = "Century", size = 14),
        plot.title = element_text(hjust = 0, size = 14),
        axis.text.x = element_blank())


df_long <- df %>% pivot_longer(contains("loss"), "type", values_to = "loss")

mean_profile <- df_long %>% group_by(epoch, type) %>%
  summarise(loss = mean(loss)) %>%
  mutate(experiment = 0) %>%
  select(experiment, everything())

p3 <- df_long %>%
  filter(epoch <= max_epochs) %>%
  # mutate(experiment = experiment + 1, epoch = epoch + 1) %>%
  ggplot(aes(epoch, loss, color = type, group = interaction(experiment, type))) +
  geom_line(alpha = 0.4) +
  geom_line(data = mean_profile %>% filter(epoch <= max_epochs), lwd = 1.2) +
  # geom_hline(yintercept = 0, color = "black", lty = 3, lwd = 1.2) +
  # scale_y_log10() +
  # scale_y_continuous(breaks = 10^c(-1:1), limits = c(0, max_y)) +
  scale_x_continuous(breaks = seq(0, max_epochs, 10)) +
  scale_color_manual(labels = c("train", "val"), values = c("yellow", "purple")) +
  labs(y = NULL, color = NULL, title = "Loss") +
  theme_bw() +
  theme(legend.position = "left",
        text = element_text(family = "Century", size = 14),
        plot.title = element_text(hjust = 0, size = 14))

p <- p1 / p2 / p3 
p + plot_annotation(title = "Five categorical, g(Z) = Z",
                    caption = expression(paste("n = 100,000; ", " q = ", 1000, ", ",
                                               sigma[b[1]]^2, " = ", 1, ", ",
                                               sigma[b[2]]^2, " = ", 2, ", ",
                                               sigma[b[3]]^2, " = ", 3, ", ",
                                               sigma[b[4]]^2, " = ", 4, ", ",
                                               sigma[b[5]]^2, " = ", 5, ", ")),
                    theme = theme(text = element_text(family = "Century", size = 16),
                                  plot.title = element_text(hjust = 0.5)))
##############

df <- read_csv("~/res_params_categ5_gZ.csv")
# df <- df %>% slice(248:n())
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

max_epochs <- 100
min_y <- -5
max_y <- 5
q <- 1000

p2 <- df_long %>%
  filter(epoch <= max_epochs) %>%
  ggplot(aes(epoch, grad, color = sig2, group = interaction(experiment, sig2))) +
  geom_line(alpha = 0.4) +
  geom_line(data = mean_profile %>% filter(epoch <= max_epochs), lwd = 1.2) +
  geom_hline(yintercept = 0, color = "black", lty = 3, lwd = 1.2) +
  # scale_y_continuous(breaks = c(-15, -10, -5, 0, 5, 10), limits = c(min_y, max_y)) +
  scale_x_continuous(breaks = seq(0, max_epochs, 10)) +
  scale_color_discrete(labels = c(
    expression(paste("  ", nabla, sigma[b[1]]^2)),
    expression(paste("  ", nabla, sigma[b[2]]^2)),
    expression(paste("  ", nabla, sigma[b[3]]^2)),
    expression(paste("  ", nabla, sigma[b[4]]^2)),
    expression(paste("  ", nabla, sigma[b[5]]^2)))) +
  labs(y = NULL, color = NULL, x = NULL, title = "Gradients") +
  theme_bw() +
  theme(legend.position = "left",
        text = element_text(family = "Century", size = 14),
        plot.title = element_text(hjust = 0, size = 14),
        axis.text.x = element_blank())


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

# max_epochs <- 30
max_y <- 12

p1 <- df_long %>%
  filter(epoch <= max_epochs, sig2 != "sig2e_est") %>%
  # mutate(experiment = experiment + 1, epoch = epoch + 1) %>%
  ggplot(aes(epoch, est, color = sig2, group = interaction(experiment, sig2))) +
  geom_line(alpha = 0.4) +
  geom_line(data = mean_profile %>% filter(epoch <= max_epochs, sig2 != "sig2e_est"), lwd = 1.2) +
  # geom_hline(yintercept = 10.0, color = "blue", lty = 3, lwd = 1.2) +
  geom_hline(yintercept = 1.0, color = "#F8766D", lty = 3, lwd = 1.2) +
  geom_hline(yintercept = 2.0, color = "#A3A500", lty = 3, lwd = 1.2) +
  geom_hline(yintercept = 3.0, color = "#00BF7D", lty = 3, lwd = 1.2) +
  geom_hline(yintercept = 4.0, color = "#00B0F6", lty = 3, lwd = 1.2) +
  geom_hline(yintercept = 5.0, color = "#E76BF3", lty = 3, lwd = 1.2) +
  # scale_y_log10() +
  scale_y_continuous(breaks = c(-1, 6)) +
  scale_x_continuous(breaks = seq(0, max_epochs, 10)) +
  scale_color_discrete(labels = c(
    expression(paste("  ", nabla, sigma[b[1]]^2)),
    expression(paste("  ", nabla, sigma[b[2]]^2)),
    expression(paste("  ", nabla, sigma[b[3]]^2)),
    expression(paste("  ", nabla, sigma[b[4]]^2)),
    expression(paste("  ", nabla, sigma[b[5]]^2)))) +
  labs(y = NULL, color = NULL, x = NULL,
       title = "Estimates") +
  theme_bw() +
  theme(legend.position = "left",
        text = element_text(family = "Century", size = 14),
        plot.title = element_text(hjust = 0, size = 14),
        axis.text.x = element_blank())


df_long <- df %>% pivot_longer(contains("loss"), "type", values_to = "loss")

mean_profile <- df_long %>% group_by(epoch, type) %>%
  summarise(loss = mean(loss)) %>%
  mutate(experiment = 0) %>%
  select(experiment, everything())

p3 <- df_long %>%
  filter(epoch <= max_epochs) %>%
  # mutate(experiment = experiment + 1, epoch = epoch + 1) %>%
  ggplot(aes(epoch, loss, color = type, group = interaction(experiment, type))) +
  geom_line(alpha = 0.4) +
  geom_line(data = mean_profile %>% filter(epoch <= max_epochs), lwd = 1.2) +
  # geom_hline(yintercept = 0, color = "black", lty = 3, lwd = 1.2) +
  # scale_y_log10() +
  # scale_y_continuous(breaks = 10^c(-1:1), limits = c(0, max_y)) +
  scale_x_continuous(breaks = seq(0, max_epochs, 10)) +
  scale_color_manual(labels = c("train", "val"), values = c("yellow", "purple")) +
  labs(y = NULL, color = NULL, title = "Loss") +
  theme_bw() +
  theme(legend.position = "left",
        text = element_text(family = "Century", size = 14),
        plot.title = element_text(hjust = 0, size = 14))

p <- p1 / p2 / p3 
p + plot_annotation(title = "Five categorical, g(Z) = ZW",
                    caption = expression(paste("n = 100,000; ", " q = ", 1000, ", ",
                                               sigma[b[1]]^2, " = ", 1, ", ",
                                               sigma[b[2]]^2, " = ", 2, ", ",
                                               sigma[b[3]]^2, " = ", 3, ", ",
                                               sigma[b[4]]^2, " = ", 4, ", ",
                                               sigma[b[5]]^2, " = ", 5, ", ")),
                    theme = theme(text = element_text(family = "Century", size = 16),
                                  plot.title = element_text(hjust = 0.5)))
