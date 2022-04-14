library(tidyverse)
library(lubridate)

bd <- as_tibble(data.table::fread("ukb_bp.tab", header=TRUE, sep="\t"))

bd$id = 1:nrow(bd)

bd$month_birth <- bd$f.52.0.0
bd$date_assess <- as_date(bd$f.53.0.0)
bd$date_assess1 <- as_date(bd$f.53.1.0)
bd$date_assess2 <- as_date(bd$f.53.2.0)
bd$date_assess3 <- as_date(bd$f.53.3.0)
bd$month_attend <- as_date(bd$f.55.0.0)

bd %>%
  mutate(n_dates = rowSums(across(date_assess:date_assess3,~{!is.na(.x)}))) %>%
  count(n_dates)

hist(bd$f.34.0.0)
sum(is.na(bd$f.34.0.0))
bd$age <- as.numeric(difftime(bd$date_assess, ymd(bd$f.34.0.0, truncated = 2L), units = "weeks")) / 52.25
bd$age1 <- as.numeric(difftime(bd$date_assess1, ymd(bd$f.34.0.0, truncated = 2L), units = "weeks")) / 52.25
bd$age2 <- as.numeric(difftime(bd$date_assess2, ymd(bd$f.34.0.0, truncated = 2L), units = "weeks")) / 52.25
bd$age3 <- as.numeric(difftime(bd$date_assess3, ymd(bd$f.34.0.0, truncated = 2L), units = "weeks")) / 52.25

bd %>% select(starts_with("f.31."))
bd$gender <- bd$f.31.0.0
bd$gender1 <- ifelse(!is.na(bd$age1), bd$gender, NA)
bd$gender2 <- ifelse(!is.na(bd$age2), bd$gender, NA)
bd$gender3 <- ifelse(!is.na(bd$age3), bd$gender, NA)

hist(bd$f.21001.0.0)
sum(is.na(bd$f.21001.0.0))
bd$bmi <- bd$f.21001.0.0
bd$bmi1 <- bd$f.21001.1.0
bd$bmi2 <- bd$f.21001.2.0
bd$bmi3 <- bd$f.21001.3.0

hist(bd$f.21002.0.0)
sum(is.na(bd$f.21002.0.0))
# bd$f.21002.0.0[is.na(bd$f.21002.0.0)] <- median(bd$f.21002.0.0, na.rm = T)
bd$weight <- bd$f.21002.0.0
bd$weight1 <- bd$f.21002.1.0
bd$weight2 <- bd$f.21002.2.0
bd$weight3 <- bd$f.21002.3.0

# bd$f.50.0.0[is.na(bd$f.50.0.0)] <- median(bd$f.50.0.0, na.rm = T)
bd$height <- bd$f.50.0.0
bd$height1 <- bd$f.50.1.0
bd$height2 <- bd$f.50.2.0
bd$height3 <- bd$f.50.3.0

bd$sbp <- bd$f.4080.0.0
bd$sbp1 <- bd$f.4080.1.0
bd$sbp2 <- bd$f.4080.2.0
bd$sbp3 <- bd$f.4080.3.0

bd$dbp <- bd$f.4079.0.0
bd$dbp1 <- bd$f.4079.1.0
bd$dbp2 <- bd$f.4079.2.0
bd$dbp3 <- bd$f.4079.3.0

add_fs <- function(v, name, nas, dflt) {
  bd[[v]][bd[[v]] %in% nas] <<- dflt
  bd[[v]][is.na(bd[[v]])] <<- dflt
  bd[[name]] <<- bd[[v]]
}

add_fs_n <- function(i, v, name, nas, dflt = 0) {
  add_fs(str_c("f.", v, ".", i, ".0"),
         str_c(name, ifelse(i == 0, "", i)), nas, dflt)
}

elong <- function(v) {
  pivot_longer(bd %>% select(id, starts_with(v)),
               cols = starts_with(v),
               names_to = "assess",
               values_to = v,
               values_drop_na = TRUE) %>%
    mutate(assess = case_when(
      assess == v ~ 0, assess == str_c(v, 1) ~ 1,
      assess == str_c(v, 2) ~ 2, assess == str_c(v, 3) ~ 3
    ))
}

walk(0:3, ~{add_fs_n(.x, 924, "walking_pace", c(-7, -3), 0)})
walk(0:3, ~{add_fs_n(.x, 943, "stair_climbing", c(-3, -1), 0)})
walk(0:3, ~{add_fs_n(.x, 1170, "get_up_morning", c(-3, -1), 0)})
walk(0:3, ~{add_fs_n(.x, 1180, "morning_evening", c(-3, -1), 0)})
walk(0:3, ~{add_fs_n(.x, 1190, "nap", c(-3, -1), 0)})
walk(0:3, ~{add_fs_n(.x, 1200, "sleepiness", c(-3, -1), 0)})
walk(0:3, ~{add_fs_n(.x, 1210, "snoring", c(-3, -1), 0)})
walk(0:3, ~{add_fs_n(.x, 1220, "dozing", c(-3, -1), -1)})
walk(0:3, ~{add_fs_n(.x, 1239, "tobacco", c(-3, -1), -1)})

walk(0:3, ~{add_fs_n(.x, 1329, "intake_fish_oily", c(), -3)})
walk(0:3, ~{add_fs_n(.x, 1339, "intake_fish_nonoily", c(), -3)})
walk(0:3, ~{add_fs_n(.x, 1349, "intake_meat", c(), -3)})
walk(0:3, ~{add_fs_n(.x, 1359, "intake_poultry", c(), -3)})
walk(0:3, ~{add_fs_n(.x, 1369, "intake_beef", c(), -3)})
walk(0:3, ~{add_fs_n(.x, 1379, "intake_lamb", c(), -3)})
walk(0:3, ~{add_fs_n(.x, 1389, "intake_pork", c(), -3)})
walk(0:3, ~{add_fs_n(.x, 1518, "hot_drink", c(-3, -2), 0)})
walk(0:3, ~{add_fs_n(.x, 1558, "intake_alcohol", c(-3, -2), 0)})

bd$f.1717.3.0 <- bd$f.1717.2.0
walk(0:3, ~{add_fs_n(.x, 1717, "skin_color", c(-3, -1), 0)})
walk(0:3, ~{add_fs_n(.x, 1757, "facial_aging", c(-3, -1), 0)})
walk(0:3, ~{add_fs_n(.x, 1797, "alive_father", c(-3, -1), -1)})
walk(0:3, ~{add_fs_n(.x, 1835, "alive_mother", c(-3, -1), -1)})
walk(0:3, ~{add_fs_n(.x, 20116, "smoking", c(-3, -1), -1)})

walk(0:3, ~{add_fs_n(.x, 134, "no_cancers", c(), 0)})
walk(0:3, ~{add_fs_n(.x, 135, "no_illnesses", c(), 0)})
walk(0:3, ~{add_fs_n(.x, 136, "no_operations", c(), 0)})
walk(0:3, ~{add_fs_n(.x, 137, "no_treatments", c(), 0)})

bd$f.189.0.0[is.na(bd$f.189.0.0)] <- median(bd$f.189.0.0, na.rm = T)
bd$f.189.1.0 <- bd$f.189.0.0
bd$f.189.2.0 <- bd$f.189.0.0
bd$f.189.3.0 <- bd$f.189.0.0
walk(0:3, ~{add_fs_n(.x, 189, "townsend", c(), median(bd$f.189.0.0, na.rm = T))})

walk(0:3, ~{add_fs_n(.x, 864, "no_days_walk", c(), -3)})
walk(0:3, ~{add_fs_n(.x, 884, "no_days_moderate_pa", c(), -3)})
walk(0:3, ~{add_fs_n(.x, 904, "no_days_vigorous_pa", c(), -3)})
walk(0:3, ~{add_fs_n(.x, 1050, "time_out_summer", c(), median(bd$f.1050.0.0, na.rm = T))})
walk(0:3, ~{add_fs_n(.x, 1060, "time_out_winter", c(), median(bd$f.1060.0.0, na.rm = T))})
walk(0:3, ~{add_fs_n(.x, 1070, "time_out_tv", c(), median(bd$f.1070.0.0, na.rm = T))})
walk(0:3, ~{add_fs_n(.x, 1080, "time_computer", c(), median(bd$f.1080.0.0, na.rm = T))})
walk(0:3, ~{add_fs_n(.x, 1090, "time_driving", c(), median(bd$f.1090.0.0, na.rm = T))})
walk(0:3, ~{add_fs_n(.x, 1160, "time_sleep", c(), median(bd$f.1160.0.0, na.rm = T))})
walk(0:3, ~{add_fs_n(.x, 1289, "intake_veg_cooked", c(), median(bd$f.1289.0.0, na.rm = T))})
walk(0:3, ~{add_fs_n(.x, 1299, "intake_veg_raw", c(), median(bd$f.1299.0.0, na.rm = T))})
walk(0:3, ~{add_fs_n(.x, 1309, "intake_fruit_fresh", c(), median(bd$f.1309.0.0, na.rm = T))})
walk(0:3, ~{add_fs_n(.x, 1319, "intake_fruit_dried", c(), median(bd$f.1319.0.0, na.rm = T))})
walk(0:3, ~{add_fs_n(.x, 1438, "intake_bread", c(), median(bd$f.1438.0.0, na.rm = T))})
walk(0:3, ~{add_fs_n(.x, 1458, "intake_cereal", c(), median(bd$f.1458.0.0, na.rm = T))})
walk(0:3, ~{add_fs_n(.x, 1488, "intake_tea", c(), median(bd$f.1488.0.0, na.rm = T))})
walk(0:3, ~{add_fs_n(.x, 1528, "intake_water", c(), median(bd$f.1528.0.0, na.rm = T))})

sbp_df <- elong("sbp")
dbp_df <- elong("dbp")
age_df <- elong("age")
gender_df <- elong("gender")
bmi_df <- elong("bmi")
weight_df <- elong("weight")
height_df <- elong("height")
walking_pace_df <- elong("walking_pace")
stair_climbing_df <- elong("stair_climbing")
get_up_morning_df <- elong("get_up_morning")
morning_evening_df <- elong("morning_evening")
nap_df <- elong("nap")
sleepiness_df <- elong("sleepiness")
snoring_df <- elong("snoring")
dozing_df <- elong("dozing")
tobacco_df <- elong("tobacco")

intake_fish_oily_df <- elong("intake_fish_oily")
intake_fish_nonoily_df <- elong("intake_fish_nonoily")
intake_meat_df <- elong("intake_meat")
intake_poultry_df <- elong("intake_poultry")
intake_beef_df <- elong("intake_beef")
intake_lamb_df <- elong("intake_lamb")
intake_pork_df <- elong("intake_pork")
hot_drink_df <- elong("hot_drink")
intake_alcohol_df <- elong("intake_alcohol")

skin_color_df <- elong("skin_color")
facial_aging_df <- elong("facial_aging")
alive_father_df <- elong("alive_father")
alive_mother_df <- elong("alive_mother")
smoking_df <- elong("smoking")

no_cancers_df <- elong("no_cancers")
no_illnesses_df <- elong("no_illnesses")
no_operations_df <- elong("no_operations")
no_treatments_df <- elong("no_treatments")
townsend_df <- elong("townsend")

no_days_walk_df <- elong("no_days_walk")
no_days_moderate_pa_df <- elong("no_days_moderate_pa")
no_days_vigorous_pa_df <- elong("no_days_vigorous_pa")
time_out_summer_df <- elong("time_out_summer")
time_out_winter_df <- elong("time_out_winter")
time_out_tv_df <- elong("time_out_tv")
time_computer_df <- elong("time_computer")
time_driving_df <- elong("time_driving")
time_sleep_df <- elong("time_sleep")
intake_veg_cooked_df <- elong("intake_veg_cooked")
intake_veg_raw_df <- elong("intake_veg_raw")
intake_fruit_fresh_df <- elong("intake_fruit_fresh")
intake_fruit_dried_df <- elong("intake_fruit_dried")
intake_bread_df <- elong("intake_bread")
intake_cereal_df <- elong("intake_cereal")
intake_tea_df <- elong("intake_tea")
intake_water_df <- elong("intake_water")

res <- reduce(list(sbp_df, age_df, gender_df, bmi_df, weight_df, height_df,
                   walking_pace_df,
                   stair_climbing_df,
                   get_up_morning_df,
                   morning_evening_df,
                   nap_df,
                   sleepiness_df,
                   snoring_df,
                   dozing_df,
                   tobacco_df,
                   intake_fish_oily_df,
                   intake_fish_nonoily_df,
                   intake_meat_df,
                   intake_poultry_df,
                   intake_beef_df,
                   intake_lamb_df,
                   intake_pork_df,
                   hot_drink_df,
                   intake_alcohol_df,
                   skin_color_df,
                   facial_aging_df,
                   alive_father_df,
                   alive_mother_df,
                   smoking_df,
                   no_cancers_df,
                   no_illnesses_df,
                   no_operations_df,
                   no_treatments_df,
                   townsend_df,
                   no_days_walk_df,
                   no_days_moderate_pa_df,
                   no_days_vigorous_pa_df,
                   time_out_summer_df,
                   time_out_winter_df,
                   time_out_tv_df,
                   time_computer_df,
                   time_driving_df,
                   time_sleep_df,
                   intake_veg_cooked_df,
                   intake_veg_raw_df,
                   intake_fruit_fresh_df,
                   intake_fruit_dried_df,
                   intake_bread_df,
                   intake_cereal_df,
                   intake_tea_df,
                   intake_water_df
),
inner_join, by = c("id", "assess"))

dim(res)

old_id <- sort(unique(res$id))
new_id <- 0:(max(old_id) - 1)

res$id <- new_id[match(res$id, old_id, nomatch = NA)]

res %>%
  count(id, sort = T)

ggplot(res %>% sample_n(10000), aes(age, sbp)) +
  geom_point(color = "red", alpha = 0.5) +
  geom_smooth(method = "loess") +
  theme_light()

ggplot(res %>% sample_n(50000), aes(age, sbp, group = factor(id))) +
  geom_line(color = "red", alpha = 0.5) +
  # geom_smooth(method = "loess") +
  theme_light()

res %>% write_csv("~/ukb_sbp_longitudinal.csv")