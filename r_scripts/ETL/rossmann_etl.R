library(tidyverse)

# Rossmann Store Sales dataset from Kaggle: https://www.kaggle.com/competitions/rossmann-store-sales/
rossmann <- read_csv("train.csv.zip")
ross_stores <- read_csv("store.csv")

# OHE-ing stores data and imputing missing data
ross_stores %>% count(StoreType)
df_store_type <- as_tibble(model.matrix(~0 + ross_stores$StoreType))
colnames(df_store_type) <- str_c("storetype_", letters[1:4])

ross_stores %>% count(Assortment)
df_store_assort <- as_tibble(model.matrix(~0 + ross_stores$Assortment))
colnames(df_store_assort) <- str_c("storeassort_", letters[1:3])

summary(ross_stores$CompetitionDistance)
ross_stores$CompetitionDistance[is.na(ross_stores$CompetitionDistance)] <- median(ross_stores$CompetitionDistance, na.rm = T)

summary(ross_stores$CompetitionOpenSinceYear)
ross_stores$CompetitionOpenSinceYear[is.na(ross_stores$CompetitionOpenSinceYear)] <- median(ross_stores$CompetitionOpenSinceYear, na.rm = T)

summary(ross_stores$CompetitionOpenSinceMonth)
ross_stores$CompetitionOpenSinceMonth[is.na(ross_stores$CompetitionOpenSinceMonth)] <- median(ross_stores$CompetitionOpenSinceMonth, na.rm = T)

count(ross_stores, Promo2)
summary(ross_stores$Promo2SinceYear)
ross_stores$Promo2SinceYear[is.na(ross_stores$Promo2SinceYear)] <- median(ross_stores$Promo2SinceYear, na.rm = T)
summary(ross_stores$Promo2SinceWeek)
ross_stores$Promo2SinceWeek[is.na(ross_stores$Promo2SinceWeek)] <- median(ross_stores$Promo2SinceWeek, na.rm = T)

ross_stores <- ross_stores %>% mutate(PromoInterval = case_when(
  PromoInterval == "Jan,Apr,Jul,Oct" ~ 1,
  PromoInterval == "Feb,May,Aug,Nov" ~ 2,
  PromoInterval == "Mar,Jun,Sept,Dec" ~ 3,
  TRUE ~ 0
))

ross_stores <- ross_stores %>%
  select(-StoreType, -Assortment) %>%
  bind_cols(df_store_type, df_store_assort)
  

# Rossmann main dataset
rossmann %>%
  filter(Store <= 5) %>%
  ggplot(aes(Date, Sales, col = factor(Store))) +
  geom_line()

count(rossmann, StateHoliday)
df_state_holiday <- as_tibble(model.matrix(~0 + rossmann$StateHoliday))
colnames(df_state_holiday) <- c("holiday_0", "holiday_a", "holiday_b", "holiday_c")

summary(rossmann$Sales)

rossmann <- rossmann %>%
  bind_cols(df_state_holiday) %>%
  mutate(
    year = lubridate::year(Date),
    month = lubridate::month(Date)) %>%
  group_by(year, month, Store) %>%
  summarise(across(c(Sales, Open, Promo, SchoolHoliday,
                     starts_with("holiday")), sum)) %>%
  ungroup() %>%
  mutate(date = lubridate::make_date(year, month, 1),
         Sales  = Sales / 100000,
          t = as.numeric(date - as.Date("2013-01-01"))) %>%
  mutate(t = t / max(.$t))

# Joining and sinking
rossmann %>%
  inner_join(ross_stores, by = "Store") %>%
  write_csv("rossmann.csv")
