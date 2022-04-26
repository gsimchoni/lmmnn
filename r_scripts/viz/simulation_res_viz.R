library(tidyverse)
library(patchwork)

library(extrafont)
font_import()
loadfonts(device="win") 

res <- read_csv("results/res_NL_100K_Z_n5.csv")

exp_type_label_list <- c(Ignore = "ignore", OHE = "ohe",
                         Embeddings = "embed", lme4 = "lme4",
                         MeNets = "menet", LMMNN = "lmm")
q_label_list = c(
  "q == 100" = "100",
  "q == 1000" = "1000",
  "q == 10000" = "10000"
)
sig2b_label_list = c(
  "sigma[b]^2 == 0.1" = "0.1",
  "sigma[b]^2 == 1" = "1",
  "sigma[b]^2 == 10" = "10"
)
p <- 
  res %>%
  group_by(sig2b, q, exp_type) %>%
  summarise(MSE = mean(mse), se = sd(mse) / sqrt(5)) %>%
  ungroup() %>%
  mutate(exp_type = fct_relevel(exp_type, c("ignore", "ohe", "embed", "lme4", "menet", "lmm")),
         exp_type = fct_recode(exp_type,
                               !!!exp_type_label_list),
         q = fct_recode(as_factor(q), !!!q_label_list),
         sig2b = fct_recode(as_factor(sig2b), !!!sig2b_label_list)) %>%
  ggplot(aes(exp_type, MSE, fill = factor(ifelse(exp_type == "LMMNN", "highlighted", "normal")))) +
  geom_errorbar(aes(x = exp_type, ymin = MSE - se, ymax = MSE + se), width = 0.25) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values=c("red", "grey50")) +
  facet_grid(sig2b ~ q, scales = "free", labeller = label_parsed) +
  labs(x = NULL, y = "Mean Test MSE") +
  guides(fill = FALSE) +
  theme_bw() +
  theme(text = element_text(family = "Century", size=20),
        axis.text.y = element_text(size = 12),
        axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("res_NL_100K_Z_n5.png", p, device = "png", width = 14, height = 7, dpi = 300)

# GLMM

res <- read_csv("results/res_glmm_nGQ5_iterations.csv")

exp_type_label_list <- c(Ignore = "ignore", OHE = "ohe",
                         Embeddings = "embed", LMMNN = "lmm")
q_label_list = c(
  "q == 100" = "100",
  "q == 1000" = "1000",
  "q == 10000" = "10000"
)
sig2b_label_list = c(
  "sigma[b]^2 == 0.1" = "0.1",
  "sigma[b]^2 == 1" = "1",
  "sigma[b]^2 == 10" = "10"
)
p <- 
  res %>%
  mutate(sig2b = sig2b0, q = q0) %>%
  group_by(sig2b, q, exp_type) %>%
  summarise(AUC = mean(auc), se = sd(auc) / sqrt(5)) %>%
  ungroup() %>%
  mutate(exp_type = fct_relevel(exp_type, c("ignore", "ohe", "embed", "lmm")),
         exp_type = fct_recode(exp_type,
                               !!!exp_type_label_list),
         q = fct_recode(as_factor(q), !!!q_label_list),
         sig2b = fct_recode(as_factor(sig2b), !!!sig2b_label_list)) %>%
  ggplot(aes(exp_type, AUC, fill = factor(ifelse(exp_type == "LMMNN", "highlighted", "normal")))) +
  geom_errorbar(aes(x = exp_type, ymin = AUC - se, ymax = AUC + se), width = 0.25) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values=c("red", "grey50")) +
  facet_grid(sig2b ~ q, labeller = label_parsed) +
  labs(x = NULL, y = "Mean Test AUC") +
  coord_cartesian(ylim = c(0.5, 1)) +
  guides(fill = FALSE) +
  theme_bw() +
  theme(text = element_text(family = "Century", size=20),
        axis.text.y = element_text(size = 12),
        axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("res_GLMM_100K_Z_n5.png", p, device = "png", width = 14, height = 7, dpi = 300)

# Real data results

res <- tribble(
  ~Dataset, ~exp_type, ~MSE, ~se,
  "UKB PA", "Ignore", 0.812, 0.008,
  "UKB PA", "OHE", 0.816, 0.009,
  "UKB PA", "Embeddings", 0.817, 0.01,
  "UKB PA", "LMMNN", 0.809, 0.008,
  "Drugs", "Ignore", 2.74, 0.032,
  "Drugs", "OHE", 2.77, 0.005,
  "Drugs", "Embeddings", 2.72, 0.051,
  "Drugs", "LMMNN", 2.66, 0.006,
  # "CelebA noseX", "Ignore", 1.68, 0.05,
  # "CelebA noseX", "OHE", 0.816, 0.009,
  # "CelebA noseX", "Embeddings", 3.6, 0.3,
  # "CelebA noseX", "LMMNN", 1.54, 0.07,
  "CelebA noseY", "Ignore", 1.64, 0.09,
  "CelebA noseY", "OHE", 0, NA,
  "CelebA noseY", "Embeddings", 2.5, 0.2,
  "CelebA noseY", "LMMNN", 1.39, 0.04,
  "Airbnb", "Ignore", 0.156, 0.002,
  "Airbnb", "OHE", 0, NA,
  "Airbnb", "Embeddings", 0.158, 0.003,
  "Airbnb", "LMMNN", 0.142, 0.002
)

res %>%
  mutate(exp_type = fct_relevel(exp_type, c("Ignore", "OHE", "Embeddings", "LMMNN"))) %>%
  ggplot(aes(exp_type, MSE, fill = factor(ifelse(exp_type == "LMMNN", "highlighted", "normal")))) +
  geom_errorbar(aes(x = exp_type, ymin = MSE - se, ymax = MSE + se), width = 0.25) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values=c("red", "grey50")) +
  facet_wrap(~ Dataset, scales = "free_y") +
  labs(x = NULL, y = "Mean Test MSE") +
  guides(fill = FALSE) +
  theme_bw() +
  theme(text = element_text(family = "Century", size=20),
        axis.text.y = element_text(size = 12),
        axis.text.x = element_text(angle = 45, hjust = 1))

p1 <- res %>%
  mutate(exp_type = fct_relevel(exp_type, c("Ignore", "OHE", "Embeddings", "LMMNN"))) %>%
  filter(Dataset == "UKB PA") %>%
  ggplot(aes(exp_type, MSE, fill = factor(ifelse(exp_type == "LMMNN", "highlighted", "normal")))) +
  geom_errorbar(aes(x = exp_type, ymin = MSE - se, ymax = MSE + se), width = 0.25) +
  geom_bar(stat = "identity") +
  coord_cartesian(ylim = c(0.8, 0.83)) +
  scale_fill_manual(values=c("red", "grey50")) +
  scale_y_continuous(breaks = c(0.8, 0.81, 0.82, 0.83)) +
  labs(x = NULL, y = NULL, title = "UKB PA") +
  guides(fill = FALSE) +
  theme_bw() +
  theme(text = element_text(family = "Century", size=16),
        axis.text.y = element_text(size = 12), axis.text.x = element_blank(),
        plot.title = element_text(hjust = 0.5))
        # axis.text.x = element_text(angle = 45, hjust = 1))

p2 <- res %>%
  mutate(exp_type = fct_relevel(exp_type, c("Ignore", "OHE", "Embeddings", "LMMNN"))) %>%
  filter(Dataset == "Drugs") %>%
  ggplot(aes(exp_type, MSE, fill = factor(ifelse(exp_type == "LMMNN", "highlighted", "normal")))) +
  geom_errorbar(aes(x = exp_type, ymin = MSE - se, ymax = MSE + se), width = 0.25) +
  geom_bar(stat = "identity") +
  coord_cartesian(ylim = c(2, 3)) +
  scale_fill_manual(values=c("red", "grey50")) +
  scale_y_continuous(breaks = c(2, 2.25, 2.5, 2.75, 3)) +
  labs(x = NULL, y = NULL, title = "Drugs") +
  guides(fill = FALSE) +
  theme_bw() +
  theme(text = element_text(family = "Century", size=16),
        axis.text.y = element_text(size = 12), axis.text.x = element_blank(),
        plot.title = element_text(hjust = 0.5))

p3 <- res %>%
  mutate(exp_type = fct_relevel(exp_type, c("Ignore", "OHE", "Embeddings", "LMMNN"))) %>%
  filter(Dataset == "CelebA noseY") %>%
  ggplot(aes(exp_type, MSE, fill = factor(ifelse(exp_type == "LMMNN", "highlighted", "normal")))) +
  geom_errorbar(aes(x = exp_type, ymin = MSE - se, ymax = MSE + se), width = 0.25) +
  geom_bar(stat = "identity") +
  coord_cartesian(ylim = c(0, 3)) +
  scale_fill_manual(values=c("red", "grey50")) +
  scale_y_continuous(breaks = c(0, 0.5, 1.0, 1.5, 2, 2.5, 3)) +
  labs(x = NULL, y = NULL, title = "CelebA noseY") +
  guides(fill = FALSE) +
  theme_bw() +
  theme(text = element_text(family = "Century", size=16),
        axis.text.y = element_text(size = 12),
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(size = 20))

p4 <- res %>%
  mutate(exp_type = fct_relevel(exp_type, c("Ignore", "OHE", "Embeddings", "LMMNN"))) %>%
  filter(Dataset == "Airbnb") %>%
  ggplot(aes(exp_type, MSE, fill = factor(ifelse(exp_type == "LMMNN", "highlighted", "normal")))) +
  geom_errorbar(aes(x = exp_type, ymin = MSE - se, ymax = MSE + se), width = 0.25) +
  geom_bar(stat = "identity") +
  coord_cartesian(ylim = c(0.1, 0.2)) +
  scale_fill_manual(values=c("red", "grey50")) +
  scale_y_continuous(breaks = c(0.1, 0.15, 0.2)) +
  labs(x = NULL, y = NULL, title = "Airbnb") +
  guides(fill = FALSE) +
  theme_bw() +
  theme(text = element_text(family = "Century", size=16),
        axis.text.y = element_text(size = 12),
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(size = 20))

p <- (p1 | p2) / (p3 | p4)


ggsave("real_data_res.png", p, device = "png", width = 14, height = 7, dpi = 300)

#### Multiple Categoricals

res <- read_csv("results/res_NL_100K_Z_n5_two_categoricals.csv")

exp_type_label_list <- c(Ignore = "ignore", OHE = "ohe",
                         Embeddings = "embed", lme4 = "lme4",
                         LMMNN = "lmm")

qs_label_list = c(
  "q[1] == 1000 ~ q[2] == 1000" = "1000_1000",
  "q[1] == 1000 ~ q[2] == 10000" = "1000_10000",
  "q[1] == 10000 ~ q[2] == 10000" = "10000_10000"
)
label_parsed(names(qs_label_list))

sig2bs_label_list = c(
  "sigma[b1]^2 == 0.5 ~ sigma[b2]^2 == 0.5" = "0.5_0.5",
  "sigma[b1]^2 == 0.5 ~ sigma[b2]^2 == 5" = "0.5_5",
  "sigma[b1]^2 == 5 ~ sigma[b2]^2 == 5" = "5_5"
)


comb_levels <- tibble::tribble(
      ~sig2bs, ~qs,
   "0.5_0.5", "1000_1000",
   "0.5_0.5", "1000_10000",
   "0.5_0.5", "10000_10000",
   "0.5_5", "1000_10000",
   "0.5_5", "10000_10000",
   "5_0.5", "1000_1000",
   "5_5", "1000_1000",
   "5_5", "1000_10000",
   "5_5", "10000_10000"
  )

p <- 
  res %>%
  group_by(sig2b0, sig2b1, q0, q1, exp_type) %>%
  summarise(MSE = mean(mse), se = sd(mse) / sqrt(5)) %>%
  ungroup() %>%
  unite("sig2bs", c(sig2b0, sig2b1)) %>%
  unite("qs", c(q0, q1)) %>%
  inner_join(comb_levels) %>%
  mutate(sig2bs = ifelse(sig2bs == "5_0.5", "0.5_5", sig2bs)) %>%
  mutate(exp_type = fct_relevel(exp_type, c("ignore", "ohe", "embed", "lme4", "lmm")),
         exp_type = fct_recode(exp_type,
                               !!!exp_type_label_list),
         qs = fct_recode(as_factor(qs), !!!qs_label_list),
         sig2bs = fct_recode(as_factor(sig2bs), !!!sig2bs_label_list)) %>%
  ggplot(aes(exp_type, MSE, fill = factor(ifelse(exp_type == "LMMNN", "highlighted", "normal")))) +
  geom_errorbar(aes(x = exp_type, ymin = MSE - se, ymax = MSE + se), width = 0.25) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values=c("red", "grey50")) +
  facet_grid(sig2bs ~ qs, scales = "free", labeller = label_parsed) +
  labs(x = NULL, y = "Mean Test MSE") +
  guides(fill = "none") +
  theme_bw() +
  theme(text = element_text(family = "Century", size=19),
        axis.text.y = element_text(size = 12),
        axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("res_NL_100K_Z_n5_two_categoricals.png", p, device = "png", width = 14, height = 7, dpi = 300)
