library(tidyverse)
library(patchwork)

library(extrafont)
font_import()
loadfonts(device="win") 

setwd("C:/Users/gsimchoni/lmmnn")

RE1 <- read_csv("data/data_for_viz/RE_sig2b1.csv")
Y1 <- read_csv("data/data_for_viz/y_test_sig2b1.csv")
RE10 <- read_csv("data/data_for_viz/RE_sig2b10.csv")
Y10 <- read_csv("data/data_for_viz/y_test_sig2b10.csv")

p_scatter_b_sig2b1 <- RE1 %>%
  ggplot(aes(b_true, b_pred)) +
  geom_point(alpha = 0.2) +
  labs(x = "", y = "Predicted RE") +
  xlim(-10, 10) +
  ylim(-10, 10) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16))

p_scatter_b_sig2b10 <- RE10 %>%
  ggplot(aes(b_true, b_pred)) +
  geom_point(alpha = 0.2) +
  labs(x = "True RE", y = "Predicted RE") +
  xlim(-10, 10) +
  ylim(-10, 10) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16))

p_hist_n_sig2b1 <- RE1 %>%
  ggplot(aes(n)) +
  geom_histogram(alpha = 0.5) +
  labs(x = "", y = "Frequency", title = expression(paste(sigma[b]^2," =1, ", q, "=1000"))) +
  xlim(0, 300) +
  ylim(0, 160) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16),
        plot.title = element_text(hjust = 0.5, size = 20))

p_hist_n_sig2b10 <- RE10 %>%
  ggplot(aes(n)) +
  geom_histogram(alpha = 0.5) +
  labs(x = "Category size", y = "Frequency", title = expression(paste(sigma[b]^2," =10, ", q, "=1000"))) +
  xlim(0, 300) +
  ylim(0, 160) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16),
        plot.title = element_text(hjust = 0.5, size = 20))

p_scatter_y_sig2b1 <- Y1 %>%
  ggplot(aes(y_true, y_pred)) +
  geom_point(alpha = 0.2) +
  labs(x = "", y = "Predicted test y") +
  xlim(-15, 15) +
  ylim(-15, 15) +
  # geom_abline(intercept = 0, slope = 1) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16))

p_scatter_y_sig2b10 <- Y10 %>%
  ggplot(aes(y_true, y_pred)) +
  geom_point(alpha = 0.2) +
  labs(x = "True test y", y = "Predicted test y") +
  xlim(-15, 15) +
  ylim(-15, 15) +
  # geom_abline(intercept = 0, slope = 1) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16))

p1 <- (p_scatter_b_sig2b1 | p_hist_n_sig2b1 | p_scatter_y_sig2b1)
p2 <- (p_scatter_b_sig2b10 | p_hist_n_sig2b10 | p_scatter_y_sig2b10)
p <- p1 / p2

ggsave("sim_viz.png", p, device = "png", width = 16, height = 8, dpi = 300)

############

ukb_job <- read_csv("data/data_for_viz/ukb_job_y_test.csv")
airbnb <- read_csv("data/data_for_viz/airbnb_y_test.csv")

ukbjob_scatter_y <- ukb_job %>%
  ggplot(aes(y_true, y_pred)) +
  geom_point(alpha = 0.2) +
  labs(x = "True test y", y = "Predicted test y") +
  xlim(-3, 4) +
  ylim(-3, 4) +
  # geom_abline(intercept = 0, slope = 1) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16))

airbnb_scatter_y <- airbnb %>%
  ggplot(aes(y_true, y_pred)) +
  geom_point(alpha = 0.2) +
  labs(x = "True test y", y = "Predicted test y") +
  xlim(2, 8) +
  ylim(2, 8) +
  # geom_abline(intercept = 0, slope = 1) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16))