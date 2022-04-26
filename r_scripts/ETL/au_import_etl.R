library(tidyverse)
library(janitor)

# AU anual commodities import data from Kaggle (by UN): 
# https://www.kaggle.com/datasets/unitednations/global-commodity-trade-statistics

# commodity data
uncommodity <- read_csv("commodity_trade_statistics_data.csv")

au_comm <- uncommodity %>%
  filter(country_or_area == "Australia", flow == "Import", commodity != "ALL COMMODITIES") %>%
  select(year, comm_code, commodity, trade_usd)

au_comm %>%
  count(year)

au_comm %>%
  count(commodity)

au_comm %>%
  count(comm_code)

au_comm %>%
  filter(commodity %in% c("Activated carbon", "Alarm clocks, non-electric",
                          "Alcoholic liqueurs nes", "Almonds in shell fresh or dried",
                          "Aluminous cement")) %>%
  ggplot(aes(year, log(trade_usd), col = commodity)) +
  geom_line()

# Other data such as temperature, child mortality, wheat yield, come from
# ourworldindata.org, e.g. https://ourworldindata.org/co2-and-other-greenhouse-gas-emissions
# temperature data
temp <- read_csv("hadcrut-surface-temperature-anomaly.csv") %>%
  filter(between(Year, 1988, 2016), Entity == "Australia")

colnames(temp)[4] <- "temp_anomaly"

temp_wide <- temp %>%
  pivot_wider(id_cols = Year, names_from = Entity, values_from = temp_anomaly, names_prefix = "temp_")
colnames(temp_wide) <- make_clean_names(colnames(temp_wide))

# child mortality
temp <- read_csv("child-mortality-around-the-world.csv") %>%
  filter(between(Year, 1988, 2016), Entity == "Australia")

colnames(temp)[4] <- "child_mortality"

child_wide <- temp %>%
  pivot_wider(id_cols = Year, names_from = Entity, values_from = child_mortality, names_prefix = "child_")

colnames(child_wide) <- make_clean_names(colnames(child_wide))

# population
temp <- read_csv("population-past-future.csv") %>%
  filter(between(Year, 1988, 2016), Entity == "Australia")
colnames(temp)[4] <- "population"

pop_wide <- temp %>%
  pivot_wider(id_cols = Year, names_from = Entity, values_from = population, names_prefix = "pop_")

colnames(pop_wide) <- make_clean_names(colnames(pop_wide))

# co2
temp <- read_csv("co-emissions-per-capita.csv") %>%
  filter(between(Year, 1988, 2016), Entity == "Australia")
colnames(temp)[4] <- "co2_emission"

co2_wide <- temp %>%
  pivot_wider(id_cols = Year, names_from = Entity, values_from = co2_emission, names_prefix = "co2_")

colnames(co2_wide) <- make_clean_names(colnames(co2_wide))

# wheat
temp <- read_csv("Attainable yields (Mueller et al. 2012).csv") %>%
  filter(between(Year, 1988, 2016), Entity == "Australia") %>%
  mutate(wheat_yield = wheat_attainable - wheat_yield_gap) %>%
  select(Entity, Year, wheat_yield)

wheat_wide <- temp %>%
  pivot_wider(id_cols = Year, names_from = Entity, values_from = wheat_yield, names_prefix = "wheat_")

colnames(wheat_wide) <- make_clean_names(colnames(wheat_wide))

# death conflict
temp <- read_csv("deaths-conflict-terrorism-per-100000.csv") %>%
  filter(between(Year, 1988, 2016), Entity == "Australia") %>%
  select(-Code)
colnames(temp)[3] <- "death_conflict"

conflict_wide <- temp %>%
  pivot_wider(id_cols = Year, names_from = Entity, values_from = death_conflict, names_prefix = "conflict_")

colnames(conflict_wide) <- make_clean_names(colnames(conflict_wide))

conflict_wide <- bind_rows(conflict_wide[1,], conflict_wide[1,], conflict_wide)
conflict_wide$year[1:2] <- c(1988, 1989)

# telephone subscribers
temp <- read_csv("fixed-landline-telephone-subscriptions-vs-gdp-per-capita.csv") %>%
  filter(between(Year, 1988, 2016), Entity == "Australia")
colnames(temp)[4] <- "tel_subscribe"

phone_wide <- temp %>%
  pivot_wider(id_cols = Year, names_from = Entity, values_from = tel_subscribe, names_prefix = "phone_")

colnames(phone_wide) <- make_clean_names(colnames(phone_wide))

# joining
au_comm_all <- au_comm %>%
  inner_join(temp_wide, by = "year") %>%
  inner_join(child_wide, by = "year") %>%
  inner_join(pop_wide, by = "year") %>%
  inner_join(co2_wide, by = "year") %>%
  inner_join(wheat_wide, by = "year") %>%
  inner_join(conflict_wide, by = "year") %>%
  inner_join(phone_wide, by = "year")

# recode commodity ID
au_comm_all <- au_comm_all %>% inner_join(
  au_comm_all %>%
    distinct(comm_code) %>%
    mutate(commodity_id = row_number() - 1),
  by = "comm_code"
) %>%
  mutate(t = (year - min(.$year)) / (max(.$year) - min(.$year)))


au_comm_all %>% write_csv("au_anual_import_commodity.csv")
