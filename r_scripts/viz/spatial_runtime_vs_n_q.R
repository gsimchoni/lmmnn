library(tidyverse)
library(patchwork)
library(extrafont)

font_import()
loadfonts(device="win") 


# Spatial increasing-n simulation viz ---------------------------------------

increase_n <- read_csv("data/data_for_viz/res_increase_n.csv")

increase_n <- increase_n %>%
  mutate(exp_type = case_when(
    exp_type == "cnn" ~ "CNN",
    exp_type == "embed" ~ "Embed.",
    exp_type == "ignore" ~ "Ignore",
    exp_type == "lmm" ~ "LMMNN",
    exp_type == "ohe" ~ "OHE",
    exp_type == "svdkl" ~ "SVDKL"
  ),
  exp_type = fct_relevel(exp_type, c("Ignore", "OHE", "Embed.", "CNN", "SVDKL", "LMMNN")))

pN <- increase_n %>%
  group_by(N, exp_type) %>%
  summarise(time = mean(time)) %>%
  ggplot(aes(log10(N), log10(time), shape = exp_type, color = exp_type)) +
  geom_point(size=4, alpha=0.8) +
  geom_line(linetype=2) +
  labs(x = "n", y = "Mean runtime (seconds)") +
  scale_x_continuous(breaks = log10(c(10000, 100000, 1000000)),
                     labels = c(expression(10^5), expression(10^6), expression(10^7))) +
  scale_y_continuous(breaks = 0:4,
                     labels = c(expression(10^0), expression(10^1), expression(10^2), expression(10^3), expression(10^4)),
                     limits = c(0, 4)) +
  scale_shape_manual("", values=c(20:15)) +
  scale_color_manual("", values=c("brown", "purple", "orange", "green", "red", "blue")) +
  guides(shape="none", color = "none") +
  theme_bw() +
  theme(text = element_text(family = "Century", size=16))


increase_q <- read_csv("data/data_for_viz/res_increase_q.csv")

increase_q <- increase_q %>%
  mutate(exp_type = case_when(
    exp_type == "cnn" ~ "CNN",
    exp_type == "embed" ~ "Embed.",
    exp_type == "ignore" ~ "Ignore",
    exp_type == "lmm" ~ "LMMNN",
    exp_type == "ohe" ~ "OHE",
    exp_type == "svdkl" ~ "SVDKL"
  ),
  exp_type = fct_relevel(exp_type, c("Ignore", "OHE", "Embed.", "CNN", "SVDKL", "LMMNN")))

pQ <- increase_q %>%
  group_by(q_spatial, exp_type) %>%
  summarise(time = mean(time)) %>%
  ggplot(aes(log10(q_spatial), log10(time), shape = exp_type, color = exp_type)) +
  geom_point(size=4, alpha=0.8) +
  geom_line(linetype=2) +
  labs(x = "q", y = "") +
  scale_x_continuous(breaks = log10(c(100, 1000, 10000)),
                     labels = c(expression(10^2), expression(10^3), expression(10^4))) +
  scale_y_continuous(breaks = 0:4, limits = c(0, 4)) +
  scale_shape_manual("", values=c(20:15)) +
  scale_color_manual("", values=c("brown", "purple", "orange", "green", "red", "blue")) +
  theme_bw() +
  theme(text = element_text(family = "Century", size=16),
        axis.text.y = element_blank())

p <- pN | pQ

ggsave("images/sim_spatial_increase_n_q.png", p, device = "png", width = 12, height = 4, dpi = 300)
